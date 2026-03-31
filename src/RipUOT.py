import numpy as np
import torch 
from utils import unbalanced_sinkhorn_log_domain
from evals import compute_lta
import copy

class RipUOT(object): 

    def __init__(self, data_s, data_t, a=None, b=None, lambd=1.0, lambd_2=None, eps=5e-3, log=False, translate=False):
        assert isinstance(data_s, torch.Tensor) and data_s.dtype == torch.float64
        assert isinstance(data_t, torch.Tensor) and data_t.dtype == torch.float64 
        "data must be a list of two torch.Tensors of type float64" 
        assert isinstance(a, torch.Tensor) and a.dtype == torch.float64 if a is not None else True
        assert isinstance(b, torch.Tensor) and b.dtype == torch.float64 if b is not None else True
        "a and b must be torch.Tensors of type float64" 
        assert bool((a > 0).all()) and a.shape[0] == data_s.shape[0] if a is not None else True
        "a must be positive and compatible with data" 
        assert bool((b > 0).all()) and b.shape[0] == data_t.shape[0] if b is not None else True
        "b must be positive and compatible with data" 
        assert (type(lambd) == float and lambd > 0) or lambd == 'inf'
        "lambd must be a positive float or 'inf', the latter is the balanced case"


        self.x_s = data_s
        self.x_t = data_t
        self.n_s = data_s.shape[0]
        self.d_s = data_s.shape[1]
        self.d_t = data_t.shape[1]
        self.n_t = data_t.shape[0]
        self.a = torch.full((data_s.shape[0],), 1 / data_s.shape[0], dtype=torch.float64) if a == None else a
        self.b = torch.full((data_t.shape[0],), 1 / data_t.shape[0], dtype=torch.float64) if b == None else b
        # a and b are the histograms of the two (positive) measures, if not provided, they are set to uniform histograms
        self.eps = eps
        self.lambd = lambd
        self.lambd_2 = lambd_2 if lambd_2 is not None else lambd

        self.M = None
        self.M_log = [] if log == True else None
        self.coupling = None
        self.coupling_log = [] if log == True else None
        self.Mip_cost = None
        # self.gw_cost = None
        self.aligned_data = None
        self.flag = True
        self.translate = translate

    def cost_ip(self, A): # inner product cost
        C_0 = torch.einsum('kh,ih,jk->ij', A, self.x_s, self.x_t)
        return - C_0

    def compute_M(self, R, k=1.0):
        R = R.to(self.x_t.device)

        PY = (R @ self.x_t).T
        A_0 = PY @ self.x_s
        return k * (A_0 / torch.norm(A_0))


    def fit(self, k_cost, max_iter, tol, max_iter_sinkhorn, tol_sinkhorn, device, verbose, log):
        x_s = self.x_s.to(device)
        x_t = self.x_t.to(device)
        a = self.a.to(device)
        b = self.b.to(device)
        eps = self.eps
        lambd = self.lambd
        lambd_2 = self.lambd_2    

        self.M = torch.ones(x_t.shape[1], x_s.shape[1], dtype=torch.float64, device=device) / max(x_s.shape[1], x_t.shape[1])
        self.M = k_cost * ( self.M / torch.norm(self.M) )

        self.coupling = torch.outer(a, b).to(device)
        tau_1 = lambd / (lambd + eps) if lambd != 'inf' else 1.0
        tau_2 = lambd_2 / (lambd_2 + eps) if lambd_2 != 'inf' else 1.0


        err = 1
        i = 1
        
        self.M_log.append(self.M.clone()) if log == True else None
        self.coupling_log.append(self.coupling.clone()) if log == True else None

        fv = None
        gv = None

        while i <= max_iter and err > tol:
            print(i) if verbose else None
            i += 1 
            M_prev = self.M.clone()
            P_prev = self.coupling.clone()

            self.Mip_cost = self.cost_ip(self.M)

            try:
                self.coupling, fv, gv = unbalanced_sinkhorn_log_domain(
                    a, b, self.Mip_cost, eps, tau_1, tau_2,
                    max_iter_sink=max_iter_sinkhorn, tol_sink=tol_sinkhorn,
                    f_init=fv, g_init=gv
                )
                # self.coupling = self.coupling.to(device)
            except Exception as e:
                if verbose: print("Sinkhorn failed:", e)
                self.flag = False
                return None, None

            if torch.isnan(self.coupling).any() or torch.isinf(self.coupling).any():
                if verbose: print("NaN/Inf detected in coupling – stopping")
                self.flag = False
                return None, None
            
            self.coupling_log.append(self.coupling.clone()) if log == True else None 
        
            self.M = self.compute_M(self.coupling, k=k_cost)
            self.M_log.append(self.M.clone()) if log == True else None

            err_M = torch.norm(self.M - M_prev)
            err_P = torch.norm(self.coupling - P_prev)
            print('error M:', err_M) if verbose else None
            print('error P:', err_P) if verbose else None
            err = torch.max(err_M, err_P)

    def entropic_map(self, eps_1, ug, ub):
        ug = ug.to(self.Mip_cost.device)
        ub = ub.to(self.Mip_cost.device)
        logits = ((ug[None, :] - self.Mip_cost) / eps_1) + ub.log()[None, :] 
        weights = torch.softmax(logits, dim=1)
        T_eps_1 = torch.matmul(weights, self.x_t)

        return T_eps_1
    
    
    def align(self, eps_1, max_iter_sinkhorn=5000, tol_sinkhorn=1e-9, verbose=False):
        ua = torch.sum(self.coupling, dim=1)
        ub = torch.sum(self.coupling, dim=0)
        ub_normal = ub / ub.sum()
        print('ub normalized', ub_normal) if verbose else None
        Mx_s = self.x_s @ (self.M).T 
        Mx_s_unique, inverse_indices_expr = torch.unique(Mx_s, dim=0, return_inverse=True)

        # Compute pushforward measure A#ua
        Mua = torch.zeros(len(Mx_s_unique), dtype=ua.dtype, device=ua.device)
        Mua = Mua.index_add(0, inverse_indices_expr.to(ua.device), ua)
        Mua_normal = Mua / Mua.sum()

        C_A = - torch.einsum('ih,jh->ij', Mx_s_unique, self.x_t)
        try:
            _, _, pot_g = unbalanced_sinkhorn_log_domain(
                Mua_normal, ub_normal, C_A, eps_1,
                max_iter_sink=max_iter_sinkhorn, tol_sink=tol_sinkhorn
            )
        except Exception as e:
            if verbose: print("Final Sinkhorn failed:", e)
            self.flag = False
            return None, None

        if torch.isnan(pot_g).any() or torch.isinf(pot_g).any():
            if verbose: print("NaN/Inf detected in final potentials – stopping")
            self.flag = False
            return None, None
        if self.flag:
            # compute the aligned source data
            X_s = self.entropic_map(eps_1, pot_g, ub)
            self.aligned_data = [X_s, self.x_t]
            return X_s, self.x_t
        else:
            print('Alignment failed due to numerical errors')
            return None, None
    
    def align_lta_log(self, labels_s, labels_t, k_cost, eps_1, max_iter=100, tol=1e-6, max_iter_sinkhorn=5000, tol_sinkhorn=1e-9, device='cpu', verbose=False, log=True):
        assert len(labels_s) == self.x_s.shape[0]
        assert len(labels_t) == self.x_t.shape[0]
        assert isinstance(labels_s, np.ndarray) and isinstance(labels_t, np.ndarray)
        assert log == True
        lta_log = []
        self.fit(k_cost=k_cost, max_iter=max_iter, tol=tol, max_iter_sinkhorn = max_iter_sinkhorn, tol_sinkhorn=tol_sinkhorn, device=device, verbose=verbose, log=log)
        n_iter = len(self.coupling_log)
        for i in range(n_iter):
            print(i)
            ua = torch.sum(self.coupling_log[i], dim=1)
            ub = torch.sum(self.coupling_log[i], dim=0)
            ub_normal = ub / ub.sum()
            Mx_s = self.x_s @ (self.M_log[i]).T 
            Mx_s_unique, inverse_indices_expr = torch.unique(Mx_s, dim=0, return_inverse=True)
            # Compute pushforward measure A#ua
            Mua = torch.zeros(len(Mx_s_unique), dtype=ua.dtype, device=ua.device)
            Mua = Mua.index_add(0, inverse_indices_expr.to(ua.device), ua)
            Mua_normal = Mua / Mua.sum()

            C_A = - torch.einsum('ih,jh->ij', Mx_s, self.x_t)
            _, _, pot_g = unbalanced_sinkhorn_log_domain(Mua_normal, ub_normal, C_A, eps_1, max_iter_sink=max_iter_sinkhorn, tol_sink=tol_sinkhorn) 
            X_s = self.entropic_map(eps_1, pot_g, ub_normal)
            lta = compute_lta(X_s.cpu().numpy(), self.x_t.cpu().numpy(), labels_s, labels_t)
            lta_log.append(lta)
            print("LTA at iteration", i, 'is', lta)

        return lta_log



    




        



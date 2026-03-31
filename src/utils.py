import torch

def cost_ip(x, y, A, k=0): # inner product cost
    # Ax = torch.matmul(x, A.T)
    # return -torch.matmul(Ax, y.T)
    # k = row_maxnorm_expr * row_maxnorm_methy
    C_0 = torch.einsum('kh,ih,jk->ij', A, x, y)
    return k - C_0

def compute_A(x, y, R, k=1.0):
    PY = (R @ y).T
    A_0 = PY @ x
    return k * (A_0 / torch.norm(A_0))

def unbalanced_sinkhorn_log_domain(α, β, M, eps, tau_1=1.0, tau_2=1.0, max_iter_sink=2000, tol_sink=1e-6, f_init=None, g_init=None, device='cpu', verbose=False):
    """
    Unbalanced Sinkhorn algorithm for entropic regularized UOT.
    
    Arguments:
    α -- Source distribution (tensor of shape (n,))
    β -- Target distribution (tensor of shape (m,))
    M -- Cost matrix (tensor of shape (n, m))
    eps -- Entropy regularization parameter
    (lambd_1, lambd_2 -- Unbalanced regularization parameters)
    tau_1, tau_2 -- lambd_1/(lambd_1 + eps), lambd_2/(lambd_2+eps), the case (1,1) corresponds to balanced Sinkhorn
    max_iter_sink -- Maximum number of iterations
    tol_sink -- Convergence tolerance
    
    Returns:
    pi -- Optimal transport plan (tensor of shape (n, m))
    f -- first Sinkhorn potential
    g -- second Sinkhorn potential
    """
    α = α.to(device)
    β = β.to(device)
    M = M.to(device)
    n, m = M.shape

    it = 1
    error = 1

    # Log-domain potentials (warm-start if provided)
    if f_init is not None:
        f = f_init.clone().to(device)
    else:
        f = torch.zeros(n, device=device, dtype=torch.float64)
        
    if g_init is not None:
        g = g_init.clone().to(device)
    else:
        g = torch.zeros(m, device=device, dtype=torch.float64)

    loga = α.log()
    logb = β.log()

    while it <= max_iter_sink and error > tol_sink:
        f_prev = f.clone()
        # update g
        g = - ( ( eps * tau_2 ) * ( ( ( f[:, None] - M ) / eps ) + loga[:,None] ).logsumexp(dim=0) )
        # update f
        f = - ( ( eps * tau_1 ) * ( ( ( g[None, :] - M ) / eps ) + logb[None,:] ).logsumexp(dim=1) )

        # compute error for f
        error = (f - f_prev).abs().max().item()
        if error <= tol_sink:
            print('Sinkhorn stopped at iteration', it, 'with error', error)
            print('Mean f:', f.mean().item(), 'Mean g:', g.mean().item())
        if torch.isfinite(f).all() == False or torch.isfinite(g).all() == False:
            return print('Warning: numerical errors at iteration', it)
        it += 1

    # Compute the optimal transport plan
    pi = ( ( ( f[:, None] + g[None, :] - M ) / eps ) + loga[:, None] + logb[None, :] ).exp()
    return pi, f, g


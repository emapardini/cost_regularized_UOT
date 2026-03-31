import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import seaborn as sns

def compute_lta(x, y, x_labels, y_labels, n=5):
    """
    Compute LTA between two numpy arrays x and y.
    
    Args:
        x: array of shape (n_1, d)
        y: array of shape (n_2, d)
        x_labels: array of shape (n_1,) with labels for x
        y_labels: array of shape (n_2,) with labels for y
        n: number of neighbors for KNN
    
    Returns:
        Scalar LTA value.
    """
    # Fit the KNN model on y
    knn = KNeighborsClassifier(n_neighbors=n, algorithm='brute')
    knn.fit(y, y_labels)

    # Predict labels for x and compute the accuracy
    x_labels_pred = knn.predict(x)
    acc = accuracy_score(x_labels_pred, x_labels)

    return acc

def plot_projection(source_proj, source_labels, target_data, target_labels, method='pca', save_pdf=False):
    """
    Plots source and target data.
    
    Args:
        source_proj: np.ndarray (n_source, d) – projected source samples
        source_labels: np.ndarray (n_source,) – labels of source samples
        target_data: np.ndarray (n_target, d) – target domain data
        target_labels: np.ndarray (n_target,) – labels of target samples
        method: 'pca' or 'umap'
    """
    all_data = np.vstack([source_proj, target_data])
    all_labels = np.concatenate([source_labels, target_labels])
    domain_ids = np.array([0]*len(source_proj) + [1]*len(target_data)) 

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Method must be 'pca' or 'umap'")

    embedding = reducer.fit_transform(all_data)
    source_embed = embedding[domain_ids == 0]
    target_embed = embedding[domain_ids == 1]

    classes = np.unique(all_labels)
    palette = sns.color_palette("tab10", len(classes))
    color_map = {cls: palette[i] for i, cls in enumerate(classes)}

    # Plot
    plt.figure(figsize=(10, 6))

    for cls in classes:
        # Source points
        idx_src = source_labels == cls
        plt.scatter(source_embed[idx_src, 0], source_embed[idx_src, 1],
                    label=f"Source {cls}", marker='o', color=color_map[cls],
                    edgecolor='black', alpha=0.7, s=50)

        # Target points
        idx_tgt = target_labels == cls
        plt.scatter(target_embed[idx_tgt, 0], target_embed[idx_tgt, 1],
                    label=f"Target {cls}", marker='o', color=color_map[cls],
                    edgecolor='black', alpha=0.7, s=50)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.5) 
    plt.tight_layout()
    plt.savefig("alignment.pdf", format="pdf", bbox_inches="tight") if save_pdf else None
    plt.show()

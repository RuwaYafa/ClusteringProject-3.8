import os
import numpy as np
import logging
import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score, calinski_harabasz_score
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

# ----------------------------------------------------------------------
# Dunn Index computation
# ----------------------------------------------------------------------
def calculate_dunn_index(X, labels, centroids, logger=None):
    logger = logger or print
    unique_labels = np.unique(labels)
    sep = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            sep = min(sep, d)

    max_diam = 0.0
    for lbl in unique_labels:
        pts = X[labels == lbl]
        if pts.size:
            center = centroids[int(lbl)]
            radius = np.max(np.linalg.norm(pts - center, axis=1))
            max_diam = max(max_diam, 2 * radius)

    if max_diam == 0:
        logger("Zero cluster diameter; Dunn set to inf")
        return float('inf')
    dunn = sep / max_diam
    # logger.info(f"Dunn Index: separation={sep:.4f}, max_diameter={max_diam:.4f}, Dunn={dunn:.4f}")
    return dunn


# ----------------------------------------------------------------------
# k-Means helper functions
# ----------------------------------------------------------------------
def initialize_centroids(X, k, method='random', logger=None):
    logger = logger or logging.getLogger(__name__)
    if method.lower() == 'random':
        idx = np.random.permutation(X.shape[0])[:k]
        centroids = X[idx]
        # logger.info(f"\nInitial centroids (random): {centroids}\n")
        return centroids
    elif method.lower() == 'farthest':
        centroids = np.zeros((k, X.shape[1]))
        centroids[0] = X[np.random.randint(X.shape[0])]
        for i in range(1, k):
            dists = np.vstack([np.linalg.norm(X - centroids[j], axis=1) for j in range(i)]).T
            idx_max = np.argmax(np.min(dists, axis=1))
            centroids[i] = X[idx_max]
        # logger.info(f"\nInitial centroids (farthest): {centroids}\n")
        return centroids
    else:
        logger.error("Unknown init method %s", method)
        raise ValueError("init method must be 'random' or 'farthest'")

def assign_clusters(X, centroids, logger=None):
    logger = logger or logging.getLogger(__name__)
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    labels = np.argmin(dists, axis=1)
    # logger.info(f"\nCluster assignments sample[:10]: {labels[:2]}\n")
    return labels

def update_centroids(X, labels, k, prev_centroids, logger=None):
    logger = logger or logging.getLogger(__name__)
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        pts = X[labels == i]
        if pts.size:
            new_centroids[i] = pts.mean(axis=0)
        else:
            new_centroids[i] = prev_centroids[i]
            logger.warning(f"Cluster {i} empty; keeping previous centroid {prev_centroids[i]}")
    # logger.info(f"\nUpdated centroids: {new_centroids}\n")
    return new_centroids

def calculate_loss(X, centroids, labels, logger=None):
    loss = 0.0
    for i in range(len(centroids)):
        pts = X[labels == i]
        if pts.size:
            loss += np.sum(np.linalg.norm(pts - centroids[i], axis=1))
    loss /= X.shape[0]
    return loss

def predict(X, centroids):
    """
    Predict the cluster for each sample in X_new based on nearest centroid.
    """
    dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    return np.argmin(dists, axis=1)

# ----------------------------------------------------------------------
# Main k-Means with unified metrics and plots
# ----------------------------------------------------------------------
def kmeans(X, k, max_iter=100, epsilon=1e-7, init_method='random', dataset_name='dataset', y_true=None,
           logger=None, suppress_plots=False):
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Running k-Means on {dataset_name}: k={k}, init={init_method}")

    centroids = initialize_centroids(X, k, method=init_method, logger=logger)
    prev_loss = np.inf
    loss_history = []

    for it in range(max_iter):
        labels = assign_clusters(X, centroids, logger=logger)
        centroids = update_centroids(X, labels, k, centroids, logger=logger)
        loss = calculate_loss(X, centroids, labels, logger=logger)
        loss_history.append(loss)

        # logger.info(f"-----------------------------------------------------------")
        # logger.info(f"Iteration {it+1}: loss={loss:.7f}")
        if abs(prev_loss - loss) < epsilon:
            logger.info(f"Converged at iter {it+1} (Δloss={abs(prev_loss-loss):.7f} < ε={epsilon})")
            break
        prev_loss = loss
    else:
        logger.info(f"Max_iter reached ({max_iter}) without convergence (Δloss={abs(prev_loss-loss):.7f} > ε={epsilon})")

    logger.info(f"Final loss: {loss:.7f}")

    output_dir = f'kmean-results/{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)

    if not suppress_plots:
        plot_convergence(loss_history, dataset_name, k, output_dir)
        plot_clusters_with_centroids(X, labels, centroids, dataset_name, k, output_dir)
        plot_clusters_with_loss(X, labels, centroids, dataset_name, k, loss, output_dir)
        plot_dunn(X, labels, centroids, dataset_name, k, output_dir)
        plot_elbow_method(X, max_k=2*k, dataset_name=dataset_name, output_dir=output_dir, init_method=init_method)

    # Metrics
    dunn = calculate_dunn_index(X, labels, centroids, logger=logger)
    silhouette = silhouette_score(X, labels) if k > 1 else None
    ch = calinski_harabasz_score(X, labels) if k > 1 else None

    msg = f"RESULT: kmeans Algorithm for {dataset_name} in k={k}"
    if silhouette is not None:
        msg += f",Silhouette={silhouette:.4f}"
    if ch is not None:
        msg += f",Calinski-Harabasz={ch:.4f}"
    msg += f",Dunn={dunn:.4f}"

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        ami = adjusted_mutual_info_score(y_true, labels)
        homo = homogeneity_score(y_true, labels)
        comp = completeness_score(y_true, labels)
        vscore = v_measure_score(y_true, labels)
        msg += (
            f",ARI={ari:.4f},AMI={ami:.4f},"
            f"Homogeneity={homo:.4f},Completeness={comp:.4f},V-Measure={vscore:.4f}"
        )

    # logger.info(f"-----------------------------------------------------------")
    logger.info(msg)

    return centroids, labels, loss

# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_convergence(loss_history, dataset_name, k, output_dir):
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.title(f"Convergence ({dataset_name}, k={k})")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    path = os.path.join(output_dir, f'convergence_{dataset_name}_k{k}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Saved convergence plot to {path}")


def plot_elbow_method(X, max_k, dataset_name, output_dir, init_method='random'):
    wcss = []
    for kk in range(1, max_k+1):
        _, lbls, _ = kmeans(
            X, kk, max_iter=50, epsilon=1e-4,
            init_method=init_method, logger=logging.getLogger(),
            dataset_name=dataset_name, suppress_plots=True
        )
        cent = np.array([X[lbls==i].mean(axis=0) if np.any(lbls==i) else np.zeros(X.shape[1]) for i in range(kk)])
        val = np.sum([np.sum(np.linalg.norm(X[lbls==i] - cent[i], axis=1)) for i in range(kk)])
        wcss.append(val)
    plt.figure()
    plt.plot(range(1, max_k+1), wcss, marker='o')
    plt.title(f"Elbow Method ({dataset_name})")
    plt.xlabel('k')
    plt.ylabel('Sum of distances')
    path = os.path.join(output_dir, f'elbow_{dataset_name}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Saved elbow plot to {path}")


def plot_clusters_with_centroids(X, labels, centroids, dataset_name, k, output_dir):
    plt.figure()
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(X)
        cent_proj = PCA(n_components=2).fit_transform(centroids)
    else:
        proj, cent_proj = X, centroids
    plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='viridis', s=50)
    plt.scatter(cent_proj[:,0], cent_proj[:,1], c='red', marker='*', s=200)
    plt.title(f"Clusters & Centroids ({dataset_name}, k={k})")
    path = os.path.join(output_dir, f'clusters_{dataset_name}_k{k}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Saved clusters plot to {path}")

def plot_clusters_with_loss(X, labels, centroids, dataset_name, k, loss, output_dir):
    plt.figure()
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(X)
        cent_proj = PCA(n_components=2).fit_transform(centroids)
    else:
        proj, cent_proj = X, centroids

    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='viridis', s=50, label='Data Points')
    plt.scatter(cent_proj[:, 0], cent_proj[:, 1], c='red', marker='*', s=200, label='Centroids')
    plt.title(f"Clusters (k={k}) with Loss={loss:.4f} — {dataset_name}")
    plt.legend()
    path = os.path.join(output_dir, f'clusters_loss_{dataset_name}_k{k}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Saved cluster+loss plot to {path}")

# just for try
def plot_dunn(X, labels, centroids, dataset_name, k, output_dir): # no need
    dunn_val = calculate_dunn_index(X, labels, centroids)
    plt.figure()
    plt.text(0.5, 0.5, f"Dunn Index: {dunn_val:.4f}",
             ha='center', va='center', fontsize=14)
    plt.title(f"Dunn Index ({dataset_name}, k={k})")
    path = os.path.join(output_dir, f'dunn_{dataset_name}_k{k}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.getLogger(__name__).info(f"Saved Dunn plot to {path}")

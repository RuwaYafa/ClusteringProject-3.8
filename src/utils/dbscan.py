
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import pairwise_distances

from kneed import KneeLocator

# ----------------------------------------------------------------------
# Elbow-based eps helper
# ----------------------------------------------------------------------
def best_eps(X, dataset_name, n_neighbors=5, logger=None):
    """
    Find a good DBSCAN `eps` via the k-distance (elbow) method.
    Returns a single float (eps). Robust to knee==None.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("Starting best_eps for %s  k=%d", dataset_name, n_neighbors)

    # k-distance curve -------------------------------------------------------
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    logger.info("points=%d  k-dist[min]=%.4f  k-dist[max]=%.4f",
                len(k_dist), k_dist[0], k_dist[-1])

    # elbow (knee) detection -------------------------------------------------
    idx  = np.arange(len(k_dist))
    knee = KneeLocator(idx, k_dist,
                       curve='convex', direction='increasing').knee
    if knee is None:
        knee = int(0.9 * len(k_dist))  # fallback to 90th percentile
        logger.warning("Knee not found; using 90th-percentile fallback.")
    eps = float(k_dist[knee])
    logger.info("knee index=%d  chosen eps=%.4f", knee, eps)

    # save k-distance plot ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(k_dist, lw=1)
    plt.axvline(knee, ls='--', label=f"elbow @ {eps:.3f}")
    plt.xlabel('Points (sorted)')
    plt.ylabel(f'{n_neighbors}-NN distance')
    plt.title(f'K-Distance – {dataset_name}')
    plt.legend()

    plot_path = f'dbscan_results/{dataset_name}/elbow_{dataset_name}_{n_neighbors}NN_eps{eps:.3f}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved k-distance plot -> %s", plot_path)

    return eps


# --- Metrics ---
def calculate_dunn_index(X, labels):
    unique_labels = [lbl for lbl in set(labels) if lbl != -1]  # exclude noise
    max_diameter = 0
    min_separation = float('inf')

    for i in unique_labels:
        Xi = X[labels == i]
        if len(Xi) > 1:
            diam = np.max(pairwise_distances(Xi))
            max_diameter = max(max_diameter, diam)

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            Xi = X[labels == unique_labels[i]]
            Xj = X[labels == unique_labels[j]]
            separation = np.min(pairwise_distances(Xi, Xj))
            min_separation = min(min_separation, separation)

    if max_diameter == 0:
        return float('inf')  # avoid divide by zero
    return min_separation / max_diameter




def dbscan_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    from scipy.optimize import linear_sum_assignment
    # Remove noise points (-1) from both true and predicted
    mask = y_pred != -1
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    if len(y_true_valid) == 0:
        return None  # all points were noise

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid)

    # Hungarian alignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned_pred = np.zeros_like(y_pred_valid)
    for i, j in zip(col_ind, row_ind):
        aligned_pred[y_pred_valid == i] = j
    acc_labels = accuracy_score(y_true_valid, aligned_pred)
    # Final accuracy
    print(acc_labels)
    return acc_labels


# ----------------------------------------------------------------------
# Main DBSCAN routine (signature unchanged)
# ----------------------------------------------------------------------
def dbscan(X, y, dataset_name,
           eps_param=None, min_samples_param=None,
           use_pca=False, logger=None):
    """
    Run DBSCAN with automatic or user-supplied parameters,
    scaling, metrics, and visualisations.
    """
    logger = logger or logging.getLogger(__name__)
    output_dir = f'dbscan_results/{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)
    logger.info("DBSCAN outputs will be saved to %s", os.path.abspath(output_dir))

    # ------------------------------------------------------------------
    # 0. Scale data for distance computations
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------------
    # 1. Determine eps & min_samples # the best parameter I tried (eps, min_sample)
    default_params = {
        'iris':              (0.6868, 4),#4
        '3gaussians_std0.6': (0.4008, 10),#3
        '3gaussians_std0.9': (0.4008, 20),#1
        'circles':           (0.1887, 7),#2
        'moons':             (0.1427, 7),#2
        'digits':            (8.5134, 10), #best neigbour k = 1
        'complex9':          (8.5134, 4), #k=9
        'vehicle':           (3.9213, 30) #3
    }
    eps         = eps_param
    min_samples = min_samples_param

    if dataset_name in default_params:
        # print('rfa')
        if eps is None:
            eps = default_params[dataset_name][0]
        if min_samples is None:
            min_samples = default_params[dataset_name][1]
    # print()
    # print('--------------------------------------------------------')
    # print(eps)
    # print(dataset_name)
    # print(min_samples)
    # print('---------------------------------------------------------')
    # Auto-eps only if caller did not set eps_param
    if eps_param is None:
        eps = best_eps(X_scaled, dataset_name, min_samples)

    logger.info("Final parameters -> eps=%.4f  min_samples=%d", eps, min_samples)

    # ------------------------------------------------------------------
    # 2. Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # ------------------------------------------------------------------
    # 3. Metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    logger.info("clusters=%d  noise=%d (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(X))

    if (y is not None and len(y) == len(labels)
            and len(np.unique(y)) > 1):
        logger.info("ARI=%.3f  Homog.=%.3f  Compl.=%.3f  AMI=%.3f",
            metrics.adjusted_rand_score(y, labels),
            metrics.homogeneity_score(y, labels),
            metrics.completeness_score(y, labels),
            metrics.adjusted_mutual_info_score(y, labels))
        logger.info(f"Dunn: {calculate_dunn_index(X, labels):.3f}")
    else:
        logger.warning("Skipping external scores – labels missing or invalid.")

    if n_clusters > 1:
        try:
            logger.info("Silhouette=%.3f",
                        metrics.silhouette_score(X_scaled, labels))
            logger.info("V-measure=%.3f",
                        metrics.v_measure_score(y, labels))
            logger.info(f"Calinski_Harabasz_Score: {calinski_harabasz_score(X, labels):.3f}")

        except Exception as e:
            logger.warning("Internal scores failed: %s", e)

    # ------------------------------------------------------------------
    # 4. 2-D visualisation (optionally PCA)
    X_vis = X_scaled
    if use_pca and X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X_scaled)
        logger.info("PCA vis -> 2D  var=%.1f%%",
                    100 * pca.explained_variance_ratio_.sum())

    # ------------------------------------------------------------------
    # 5. Plot clusters
    plt.figure(figsize=(12, 6))

    # style 1 – core vs edge
    ax1 = plt.subplot(121)
    unique_labels = set(labels)
    colours = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colours):
        if k == -1:
            col = [0, 0, 0, 1]          # noise -> black
        mask = (labels == k)
        # core
        xy = X_vis[mask & core_samples_mask]
        ax1.scatter(xy[:, 0], xy[:, 1], s=100, c=[col],
                    edgecolor='k', label=f'Cluster {k} Core')
        # edge
        xy = X_vis[mask & ~core_samples_mask]
        ax1.scatter(xy[:, 0], xy[:, 1], s=30,  c=[col],
                    edgecolor='k', label=f'Cluster {k} Edge')

    ax1.set_title(f"Core / Edge Points\nclusters={n_clusters}, noise={n_noise}")
    ax1.legend(loc='lower left', fontsize='small')

    # style 2 – circles
    ax2 = plt.subplot(122)
    scatter = ax2.scatter(X_vis[:, 0], X_vis[:, 1], c=labels,
                          cmap='viridis', s=50)
    for k in unique_labels:
        if k == -1:
            continue
        pts = X_vis[labels == k]
        if pts.size:
            centre = pts.mean(axis=0)
            ax2.add_patch(Ellipse(centre, width=eps*2, height=eps*2,
                                  fill=False, color='red', ls='--'))
    ax2.set_title(f"Cluster Boundaries  (eps={eps:.2f})")
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')

    plt.suptitle(f"DBSCAN • {dataset_name}  (eps={eps}, min_s={min_samples})"
                 + (" + PCA vis" if use_pca else ""))
    plt.tight_layout()

    fname = (f'dbscan_{dataset_name}_eps{eps:.3f}_min{min_samples}'
             + ('_pca' if use_pca else '') + '.png')
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved DBSCAN plot -> %s", os.path.join(output_dir, fname))

    return labels






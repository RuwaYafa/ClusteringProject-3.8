



"""
DBSCAN Analysis with Comprehensive Metrics and Visualization
- Handles both high-D and 2D data
- Compares raw vs PCA-reduced results
- Provides multiple validation metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from matplotlib.patches import Ellipse
import logging
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import pairwise_distances

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


def best_eps(X, dataset_name, n_neighbors=5, logger=None):
    """
    Find a good DBSCAN `eps` via the k-distance (elbow) method.
    Returns (eps, plot_path).
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("Starting best_eps for %s with k=%d", dataset_name, n_neighbors)

    # 1. k-distance curve -----------------------------------------------------
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    logger.info("Points=%d  k-dist[min]=%.4f  k-dist[max]=%.4f",
                len(k_dist), k_dist[0], k_dist[-1])

    # 2. Elbow detection ------------------------------------------------------
    ix   = np.arange(len(k_dist))
    knee = KneeLocator(ix, k_dist, curve='convex', direction='increasing').knee
    eps  = float(k_dist[knee])
    logger.info("knee index=%d  best eps=%.4f", knee, eps)

    # 3. Plot & save ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(k_dist, lw=1)
    plt.axvline(knee, ls='--', label=f"elbow @ {eps:.3f}")
    plt.xlabel('Points (sorted)')
    plt.ylabel(f'{n_neighbors}-NN distance')
    plt.title(f'K-Distance Curve – {dataset_name}')
    plt.legend()

    plot_path = f'../dbscan_results/elbow_{dataset_name}_{n_neighbors}NN_eps{eps:.3f}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved k-distance plot to %s", plot_path)

    return eps#, plot_path

# def best_eps(X, dataset_name, n_neighbors, logger=None): # ADDED logger argument
#     if logger is None: # Fallback logger
#         logger = logging.getLogger()
#
#     from sklearn.neighbors import NearestNeighbors
#     # import matplotlib.pyplot as plt # Keep commented out if not used globally
#
#     neigh = NearestNeighbors(n_neighbors=n_neighbors) # Use n_neighbors which is min_samples
#     nbrs = neigh.fit(X)
#     distances, indices = nbrs.kneighbors(X)
#     distances = np.sort(distances[:, -1], axis=0)
#
#     plt.figure(figsize=(10, 6)) # ADDED: Ensure new figure for plot
#     plt.plot(distances)
#     plt.xlabel('Points sorted by distance')
#     plt.ylabel(f'{n_neighbors}th nearest neighbor distance') # Use n_neighbors here
#     plt.title('K-Distance Curve')
#     plt.grid(True) # ADDED: Grid for better readability
#
#     # Save the K-Distance plot
#     plot_path = f'../dbscan_results/Elbow_{dataset_name}_distance_curve.png' # Specific path
#     os.makedirs(os.path.dirname(plot_path), exist_ok=True) # Ensure directory exists
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close() # Close the plot to free memory
#     logger.info(f"Saved K-Distance Curve to {plot_path}") # ADDED: Log save action


# DBSCAN analysis
def dbscan(X, y, dataset_name, eps_param=None, min_samples_param=None, use_pca=False, logger=None): # ADDED logger argument, renamed eps/min_samples to avoid conflict
    if logger is None: # Fallback logger
        logger = logging.getLogger()

    output_dir = '../dbscan_results' # Adjusted relative path to be consistent with main.py
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving DBSCAN visualizations to: {os.path.abspath(output_dir)}") # Added logger.info

    labels_true = y

    # Data preparation
    X_vis = X # Initialize X_vis
    if use_pca and X.shape[1] > 2:
        try:
            pca_model = PCA(n_components=2)
            X_vis = pca_model.fit_transform(X)
            logger.info(f"Reduced to {X_vis.shape[1]}D for visualization using PCA. Explained variance: {pca_model.explained_variance_ratio_.sum()*100:.1f}%") # Changed print to logger.info
        except Exception as e:
            logger.error(f"Error during PCA reduction for {dataset_name}: {e}")
            X_vis = X # Fallback to original X if PCA fails
    elif use_pca and X.shape[1] <= 2:
        logger.warning(f"PCA requested for {dataset_name}, but data already has <= 2 features ({X.shape[1]}D). Skipping PCA for visualization.")


    # --- Dataset specific parameters and best_eps call ---
    # Use parameters passed as arguments first, if they are not None
    eps = eps_param
    min_samples = min_samples_param

    # If parameters were not passed, use dataset-specific defaults

    if dataset_name == 'iris':
        eps = 0.6
        min_samples = 4
    elif dataset_name == '3gaussians_std0.6':
        eps = 0.35 #0.5~2cluster 5 #0.7 ~2cluster 20
        min_samples = 10
    elif dataset_name == '3gaussians_std0.9':
        eps = .54 #.55#0.5#1.1#.8#0.35~2
        min_samples = 20#25#30
    elif dataset_name == 'circles':
        eps = .18#.42#.54#.55
        min_samples = 7#40#60#60
    elif dataset_name == 'moons':
        eps = 0.18#0.15
        min_samples = 7#5
    elif dataset_name == 'digits':
        # eps = 8 #0.15
        min_samples = 128#5

    # ADDED: Log the effective eps and min_samples
    logger.info(f"Effective parameters for {dataset_name}: eps={eps}, min_samples={min_samples}")

    # Call best_eps for visualization (it will save the plot internally)
    eps = best_eps(X, dataset_name, min_samples, logger=logger) # Pass logger to best_eps


    # Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)

    logger.info(f"\n=== DBSCAN on {dataset_name} (eps={eps}, min_samples={min_samples}, PCA_vis={use_pca}) ===") # Changed print to logger.info
    logger.info(f"Clusters: {n_clusters} | Noise: {n_noise} ({n_noise / len(X):.1%})") # Changed print to logger.info

    # Only calculate metrics if y_true is valid and available
    if labels_true is not None and len(labels_true) == len(labels) and len(np.unique(labels_true)) > 1:
        logger.info(f"ARI: {metrics.adjusted_rand_score(y, labels):.3f}") # Changed print to logger.info
        logger.info(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}") # Changed print to logger.info
        logger.info(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}") # Changed print to logger.info
        logger.info(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}") # Changed print to logger.info
    else:
        logger.warning("True labels not provided or invalid for external metrics (ARI, Homogeneity, Completeness, AMI).")

    if n_clusters > 1:
        try:
            logger.info(f"Silhouette: {metrics.silhouette_score(X, labels):.3f}") # Changed print to logger.info
        except Exception as e:
            logger.warning(f"Could not calculate Silhouette Score: {e}")
        try:
            logger.info(f"V-measure: {metrics.v_measure_score(y, labels):.3f}") # Changed print to logger.info (requires y and labels)
        except Exception as e:
            logger.warning(f"Could not calculate V-measure Score: {e}")
    else:
        logger.info("Silhouette: N/A (requires > 1 cluster)") # Changed print to logger.info
        logger.info("V-measure: N/A (requires > 1 cluster)") # Changed print to logger.info


    # Visualization
    plt.figure(figsize=(12, 6))

    # Style 1: Core vs edge points (like StackOverflow)
    ax1 = plt.subplot(121)
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise

        class_member_mask = (labels == k)

        # Core points
        xy = X_vis[class_member_mask & core_samples_mask]
        ax1.scatter(xy[:, 0], xy[:, 1], c=[col], s=100, edgecolor='k', label=f'Cluster {k} Core')

        # Edge points
        xy = X_vis[class_member_mask & ~core_samples_mask]
        ax1.scatter(xy[:, 0], xy[:, 1], c=[col], s=30, edgecolor='k', label=f'Cluster {k} Edge')

    ax1.set_title(f"Core/Edge Points\nClusters: {n_clusters}, Noise: {n_noise}")
    ax1.legend(loc='lower left', fontsize='small') # ADDED: Legend for clarity

    # Style 2: With cluster circles (like your version)
    ax2 = plt.subplot(122)
    scatter = ax2.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, cmap='viridis', s=50)

    # Add cluster circles
    for k in unique_labels:
        if k == -1:
            continue
        cluster_points = X_vis[labels == k]
        if len(cluster_points) > 0: # Avoid error for empty clusters
            center = cluster_points.mean(axis=0)
            ax2.add_patch(Ellipse(center, width=eps * 2, height=eps * 2,
                                  fill=False, color='red', linestyle='--'))
        else:
            logger.warning(f"Cluster {k} is empty in visualization. Skipping drawing circle.") # Added warning

    ax2.set_title(f"With Cluster Boundaries\neps={eps} radius")
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')

    plt.suptitle(f"DBSCAN on {dataset_name} (eps={eps}, min_samples={min_samples})" + (" with PCA vis" if use_pca else "")) # ADDED: PCA indication
    plt.tight_layout()
    plot_filename = f'dbscan_{dataset_name}_eps{eps}_min_s{min_samples}' + ('_pca_vis' if use_pca else '') + '.png' # More descriptive filename
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    plt.close() # Close the plot to free memory
    logger.info(f"Saved DBSCAN plot to {os.path.join(output_dir, plot_filename)}") # ADDED: Log save action

    return labels

#
# # Main execution (commented out as per original)
# if __name__ == '__main__':
#     # Load and scale data
#     X_scaled, y, feature_names = load_data()
#     print("Data shape:", X_scaled.shape)
#     print("True labels:", np.unique(y))
#
#     # Run with optimal Iris parameters
#     labels_4d = run_dbscan(X_scaled, y, eps=0.6, min_samples=4,
#                            use_pca=False, dataset_name="iris_4D")
#
#     # Compare with PCA-reduced version
#     labels_2d = run_dbscan(X_scaled, y, eps=0.6, min_samples=4,
#                            use_pca=True, dataset_name="iris_2D")
#
#     # Parameter sensitivity analysis
#     print("\n=== Parameter Sensitivity ===")
#     for eps in [0.3, 0.5, 0.7]:
#         _ = run_dbscan(X_scaled, y, eps=eps, min_samples=4, use_pca=False)
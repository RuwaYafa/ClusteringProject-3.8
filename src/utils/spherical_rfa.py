"""
Spherical K-Means clustering implementation using spherecluster package
"""
import numpy as np
from spherecluster.spherical_kmeans import SphericalKMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn import metrics
from sklearn.metrics import calinski_harabasz_score


# Assuming plot_pca_projection and save_spherical_3d_pca_visualization are in src/utils/visualization.py
from src.utils.visualization import plot_pca_projection#, save_spherical_3d_pca_visualization # Import the 3D viz function


def spherical_rfa(X, y_true, dataset_name, k, logger=None): # <--- MODIFIED: Added logger argument

    if logger is None:
        # Fallback to a basic logger if none is provided (e.g., for testing directly)
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

    logger.info(f"\n--- Running Spherical K-Means for {dataset_name} (k={k}) ---")

    # --- IMPORTANT CHECK: Ensure X is valid input data ---
    if X is None or not isinstance(X, (np.ndarray, list, tuple)) or (isinstance(X, np.ndarray) and X.size == 0):
        logger.error(f"Error: Input 'X' to spherical_rfa is invalid ({type(X)}, shape: {X.shape if isinstance(X, np.ndarray) else 'N/A'}). Expected array-like data.")
        raise ValueError("Input data 'X' cannot be None or empty.")

    # L2-normalize the data
    x_normalized = normalize(X, norm='l2')

    # Ensure x_normalized is not None after the normalize call
    if x_normalized is None or (isinstance(x_normalized, np.ndarray) and x_normalized.size == 0):
        logger.error("Error: normalize() returned None or an empty array. This should not happen with sklearn.preprocessing.normalize with valid input.")
        raise RuntimeError("Normalization failed or returned None/empty array.")

    # Initialize and fit Spherical K-Means
    skm = SphericalKMeans(n_clusters=k, random_state=42)
    skm.fit(x_normalized)

    labels = skm.labels_
    centroids = skm.cluster_centers_

    # --- Clustering Metrics Calculation and Logging ---
    logger.info(f"\n--- Clustering Metrics for {dataset_name} (k={k}) ---")

    # Silhouette Score (requires > 1 cluster)
    if k > 1 and len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
        # Note: Silhouette score uses the original (or scaled, but not normalized) data X
        try:
            logger.info(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")
        except Exception as e:
            logger.warning(f"Could not calculate Silhouette Score for {dataset_name}: {e}")
    else:
        logger.info("Silhouette Coefficient: N/A (requires > 1 unique cluster or data points)")

    # Metrics that require true labels
    # Ensure y_true is not None and matches the shape of labels, and has more than 1 unique label
    if y_true is not None and len(y_true) == len(labels) and len(np.unique(y_true)) > 1:
        try:
            logger.info(f"Homogeneity: {metrics.homogeneity_score(y_true, labels):.3f}")
            logger.info(f"Completeness: {metrics.completeness_score(y_true, labels):.3f}")
            logger.info(f"V-measure: {metrics.v_measure_score(y_true, labels):.3f}")
            logger.info(f"Adjusted Rand Index: {metrics.adjusted_rand_score(y_true, labels):.3f}")
            logger.info(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(y_true, labels):.3f}")
            logger.info(f"Calinski_Harabasz_Score: {calinski_harabasz_score(X, labels):.3f}")
            logger.info(f"Dunn: {calculate_dunn_index(X, labels, centroids):.3f}")


        except Exception as e:
            logger.warning(f"Error calculating external metrics for {dataset_name}: {e}")
            logger.info("External metrics calculation skipped due to error.")
    else:
        logger.info("Metrics requiring true labels (Homogeneity, Completeness, V-measure, ARI, AMI): N/A (true labels not provided or invalid)")


    logger.info("--------------------------------------------------")
    # --- END Metrics Section ---

    # --- Saving Visualizations ---
    logger.info(f"\n--- Saving Visualizations for {dataset_name} ---")
    try:
        # Call save_spherical_visualizations (for 2D plots)
        save_spherical_visualizations(
            X, # Pass original X for PCA, as normalization changes scale
            labels,
            centroids,
            dataset_name,
            k
        )
        logger.info(f"Saved 2D visualization plots for {dataset_name}.")

        # Call the new 3D PCA visualization
        save_spherical_3d_pca_visualization(
            X, # Pass original X for PCA, as normalization changes scale
            labels,
            centroids,
            dataset_name,
            k,
            logger=logger # Pass the logger to the 3D viz function too
        )
        logger.info(f"Saved 3D PCA plot for {dataset_name}.")

    except Exception as e:
        logger.error(f"Error saving visualizations for {dataset_name}: {e}")

    return labels, centroids

def save_spherical_visualizations(X, labels, centroids, dataset_name, k):
    """
    Generates 3 specialized plots for spherical clustering:
    1. PCA projection (2D)
    2. Centroid cosine distance matrix
    3. Centroid angular relationships (radians)
    """
    # Create the directory for 2D results
    output_dir_2d = f'spherical-results/2d/{dataset_name}/' # Changed path for organization
    os.makedirs(output_dir_2d, exist_ok=True)

    # 1. PCA Projection (2D)
    fig = plot_pca_projection(X, labels, centroids, dataset_name)
    ax = fig.axes[0]
    # Removed set_sizes and set_marker here, as they should be handled in plot_pca_projection directly
    ax.set_title(f"Spherical K-Means on {dataset_name} (k={k})")
    fig.savefig(f'{output_dir_2d}/{dataset_name}_spherical_{k}_pca.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


    # Ensure centroids are not None and have expected shape for plotting
    if centroids is None or centroids.shape[0] == 0:
        # Use a logger here if this function also gets a logger argument
        print(f"Warning: No centroids to plot for {dataset_name} (k={k}). Skipping centroid plots.")
        return # Exit the function if no centroids to plot

    # 2. Centroid Cosine Distance Matrix
    plt.figure(figsize=(10, 6))
    if k == 1:
        cosine_dist = np.array([[0.0]]) # For a single cluster, the distance matrix is just [0]
        plt.imshow(cosine_dist, cmap='viridis', vmin=0, vmax=2)
        plt.text(0, 0, f"{cosine_dist[0, 0]:.2f}", ha='center', va='center', color='w')
    else:
        # np.clip handles potential floating point inaccuracies that might push values slightly outside [-1, 1]
        cosine_dist = 1 - np.dot(normalize(centroids, norm='l2'), normalize(centroids, norm='l2').T) # Ensure centroids are normalized before dot product
        plt.imshow(cosine_dist, cmap='viridis', vmin=0, vmax=2)

        # Annotate values (limit annotation for readability for larger k)
        if k <= 10:
            for i in range(k):
                for j in range(k):
                    plt.text(j, i, f"{cosine_dist[i, j]:.2f}",
                             ha='center', va='center', color='w')

    plt.colorbar(label='Cosine Distance')
    plt.xticks(range(k), [f'C{i}' for i in range(k)])
    plt.yticks(range(k), [f'C{i}' for i in range(k)])
    plt.title(f"Spherical Centroid Distances\n{dataset_name} (k={k})")
    plt.savefig(f'{output_dir_2d}/{dataset_name}_spherical_{k}_cosine_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Centroid Angular Relationships
    plt.figure(figsize=(10, 6))
    if k == 1:
        angles = np.array([[0.0]]) # For a single cluster, angle is 0
        plt.imshow(angles, cmap='coolwarm', vmin=0, vmax=np.pi)
        plt.text(0, 0, f"{angles[0, 0]:.2f}", ha='center', va='center', color='k')
    else:
        # Ensure centroids are normalized before arccos for angular distance
        # np.clip handles potential floating point inaccuracies that might push values slightly outside [-1, 1]
        angles = np.arccos(np.clip(np.dot(normalize(centroids, norm='l2'), normalize(centroids, norm='l2').T), -1, 1)) # In radians
        plt.imshow(angles, cmap='coolwarm', vmin=0, vmax=np.pi)

        # Annotate values
        if k <= 10:
            for i in range(k):
                for j in range(k):
                    plt.text(j, i, f"{angles[i, j]:.2f}",
                             ha='center', va='center', color='k')

    plt.colorbar(label='Angle (radians)')
    plt.xticks(range(k), [f'C{i}' for i in range(k)])
    plt.yticks(range(k), [f'C{i}' for i in range(k)])
    plt.title(f"Spherical Centroid Angles\n{dataset_name} (k={k})")
    plt.savefig(f'{output_dir_2d}/{dataset_name}_spherical_{k}_angle_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_spherical_3d_pca_visualization(X, labels, centroids, dataset_name, k, logger=None): # <--- MODIFIED: Added logger
    if logger is None:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

    # Create the directory for 3D results
    output_dir_3d = f'spherical-results/3d/{dataset_name}/' # Changed path for organization
    os.makedirs(output_dir_3d, exist_ok=True)

    # Determine components for PCA
    n_components = min(3, X.shape[1])

    if n_components < 2:
        logger.warning(
            f"Warning: Cannot perform 3D PCA for {dataset_name} as it only has {X.shape[1]} feature(s). Skipping 3D visualization."
        )
        return

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Transform centroids to PCA space
    if centroids.ndim == 1:
        centroids_pca = pca.transform(centroids.reshape(1, -1))
    else:
        centroids_pca = pca.transform(centroids)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2] if n_components == 3 else np.zeros(X_pca.shape[0]),
               c=labels, cmap='viridis', s=50, alpha=0.6, label='Data Points')

    # Plot centroids
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               centroids_pca[:, 2] if n_components == 3 else np.zeros(centroids_pca.shape[0]),
               c='red', s=300, marker='*', edgecolors='black', label='Centroids')

    # Set labels and title based on actual components used
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    if n_components == 3:
        ax.set_zlabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)')
    else:
        ax.set_zlabel('Principal Component 3 (N/A)')

    ax.set_title(f'3D PCA Projection of Spherical K-Means on {dataset_name} (k={k})')
    ax.legend()
    ax.grid(True)

    # Save the figure
    fig.savefig(f'{output_dir_3d}/{dataset_name}_spherical_{k}_3d_pca.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved 3D PCA plot to {output_dir_3d}/{dataset_name}_spherical_{k}_3d_pca.png")


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
    logger(f"Dunn Index: separation={sep:.4f}, max_diameter={max_diam:.4f}, Dunn={dunn:.4f}")
    return dunn

# just for try
def plot_dunn(X, labels, centroids, dataset_name, k, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dunn_val = calculate_dunn_index(X, labels, centroids)
    plt.figure()
    plt.text(0.5, 0.5, f"Dunn Index: {dunn_val:.4f}", ha='center', va='center', fontsize=14)
    plt.title(f"Dunn Index ({dataset_name}, k={k})")
    path = os.path.join(output_dir, f'dunn_{dataset_name}_k{k}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
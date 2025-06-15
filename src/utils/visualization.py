import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import combinations
# ADDED: Import setup_logging for fallback, or use logging directly if not passed
import logging

# ADDED: Import for 3D plotting if it's used elsewhere in this file
from mpl_toolkits.mplot3d import Axes3D

def plot_pairwise_features(X, labels, centroids, feature_names, dataset_name, logger=None): # ADDED logger argument
    """Visualize all feature pairs with cluster assignments."""
    if logger is None:
        logger = logging.getLogger() # Get default logger if not provided

    n_features = X.shape[1]
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    fig.suptitle(f'{dataset_name} - Pairwise Feature Visualizations', y=1.02)

    for i, j in combinations(range(n_features), 2):
        ax = axes[i, j] if n_features >= 2 else axes
        # print(ax) # Removed: No longer printing ax object
        ax.scatter(X[:, i], X[:, j], c=labels, s=50, cmap='viridis', alpha=0.6)
        ax.scatter(centroids[:, i], centroids[:, j], c='red', s=200, marker='*')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])

    # Hide redundant plots
    for i in range(n_features):
        for j in range(n_features):
            if i >= j:
                axes[i, j].axis('off')

    plt.tight_layout()
    return fig

def plot_pca_projection(X, labels, centroids, dataset_name, logger=None): # ADDED logger argument
    """Visualize clusters in PCA-reduced space."""
    if logger is None:
        logger = logging.getLogger() # Get default logger if not provided

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)

    fig, ax = plt.subplots(figsize=(10, 6))
    # MODIFIED: Changed marker for centroids as per previous discussion
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=300, marker='*') # Changed to '*' and s=300
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(f'{dataset_name} - PCA Projection')
    plt.colorbar(scatter, ax=ax)

    # Variance explained
    logger.info(f"\n--- PCA Components for {dataset_name} ---") # Changed print to logger.info
    logger.info("1st PC explains {:.1f}% variance".format(pca.explained_variance_ratio_[0]*100)) # Changed print to logger.info
    logger.info("2nd PC explains {:.1f}% variance".format(pca.explained_variance_ratio_[1]*100)) # Changed print to logger.info
    logger.info("Total explained variance: {:.1f}%".format(sum(pca.explained_variance_ratio_)*100)) # Changed print to logger.info

    return fig

def plot_tsne_projection(X, labels, centroids, dataset_name, perplexity=30, logger=None): # ADDED logger argument
    """Visualize clusters in t-SNE space."""
    if logger is None:
        logger = logging.getLogger() # Get default logger if not provided

    # Handle cases where t-SNE might not be applicable
    if X.shape[0] < perplexity * 3: # A common heuristic for t-SNE data size
        logger.warning(f"Skipping t-SNE for {dataset_name}: Not enough samples ({X.shape[0]}) for perplexity {perplexity}. Need at least 3*perplexity.")
        return None # Return None if t-SNE cannot be performed

    if X.shape[1] < 2:
        logger.warning(f"Skipping t-SNE for {dataset_name}: Data has less than 2 features. t-SNE requires at least 2 dimensions to project to.")
        return None

    # Combine data and centroids for consistent embedding
    combined = np.vstack([X, centroids])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_tsne = tsne.fit_transform(combined)

    X_tsne = combined_tsne[:-len(centroids)]
    centroids_tsne = combined_tsne[-len(centroids):]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=50, cmap='viridis', alpha=0.6)
    ax.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], c='red', s=200, marker='*')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f'{dataset_name} - t-SNE Projection')
    plt.colorbar(scatter, ax=ax)

    # For t-SNE (after fitting)
    logger.info(f"\n--- T-SNE Diagnostics for {dataset_name} ---") # Changed print to logger.info
    logger.info(f"Perplexity: {tsne.perplexity}") # Changed print to logger.info
    logger.info(f"KL Divergence: {tsne.kl_divergence_:.2f}") # Changed print to logger.info
    logger.info("Note: t-SNE components don't explain variance like PCA.") # Changed print to logger.info

    return fig

def save_visualizations(X, labels, centroids, feature_names, dataset_name, k, eps, iterations, logger=None): # ADDED logger argument

    if logger is None:
        logger = logging.getLogger() # Get default logger if not provided

    output_dir = '../kmean-results' # Adjusted relative path to be consistent with main.py
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving K-Means visualizations to: {os.path.abspath(output_dir)}") # Added logger.info

    # Only create pairwise plot for datasets with <= 4 features
    if X.shape[1] <= 4:
        try:
            fig_pairwise = plot_pairwise_features(X, labels, centroids, feature_names, dataset_name, logger=logger) # Pass logger
            fig_pairwise.savefig(f'{output_dir}/{dataset_name}_{k}_{eps}-{iterations}_pairwise.png', dpi=300, bbox_inches='tight')
            plt.close(fig_pairwise)
            logger.info(f"Saved pairwise features plot for {dataset_name}.") # Added logger.info
        except Exception as e:
            logger.error(f"Error saving pairwise features plot for {dataset_name}: {e}")
    else:
        logger.info(f"Skipping pairwise features plot for {dataset_name} as it has more than 4 features.") # Added logger.info

    # Create dimensionality reduction plots
    try:
        fig_pca = plot_pca_projection(X, labels, centroids, dataset_name, logger=logger) # Pass logger
        fig_pca.savefig(f'{output_dir}/{dataset_name}_{k}_{eps}-{iterations}_pca.png', dpi=300, bbox_inches='tight')
        plt.close(fig_pca)
        logger.info(f"Saved PCA projection plot for {dataset_name}.") # Added logger.info
    except Exception as e:
        logger.error(f"Error saving PCA projection plot for {dataset_name}: {e}")


    if len(X) <= 1000:  # t-SNE is slow for large datasets
        try:
            fig_tsne = plot_tsne_projection(X, labels, centroids, dataset_name, logger=logger) # Pass logger
            if fig_tsne is not None: # Check if t-SNE plot was actually created
                fig_tsne.savefig(f'{output_dir}/{dataset_name}_{k}_{eps}-{iterations}_tsne.png', dpi=300, bbox_inches='tight')
                plt.close(fig_tsne)
                logger.info(f"Saved t-SNE projection plot for {dataset_name}.") # Added logger.info
        except Exception as e:
            logger.error(f"Error saving t-SNE projection plot for {dataset_name}: {e}")
    else:
        logger.info(f"Skipping t-SNE plot for {dataset_name} as it has more than 1000 samples.") # Added logger.info

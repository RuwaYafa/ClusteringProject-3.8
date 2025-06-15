import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits

# ADDED: Import setup_logging from your logger utility
from src.utils.logger import setup_logging

def download_datasets(logger=None): # ADDED: Add logger as an argument, with a default of None
    """
    Downloads various datasets and saves them to the ../data directory.
    Logs progress and errors using the provided logger.
    """
    # Fallback logger if not provided (e.g., if called directly for testing)
    if logger is None:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

    # Create data directory if it doesn't exist
    data_dir = '../data_out'
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Ensured data directory exists: {os.path.abspath(data_dir)}") # Changed print to logger.info

    # # Iris dataset
    # iris = load_iris()
    # df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    # df_iris['target'] = iris.target
    # iris_path = os.path.join(data_dir, 'iris.csv')
    # df_iris.to_csv(iris_path, index=False)
    # logger.info(f"Successfully downloaded iris dataset to {iris_path}") # Changed print to logger.info

    # digit dataset
    digits = load_digits()
    # df_digits = pd.DataFrame(digits.data, columns=digits.feature_names)
    df_digits = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    df_digits['target'] = digits.target
    digits_path = os.path.join(data_dir, 'digits.csv')
    df_digits.to_csv(digits_path, index=False)
    logger.info(f"Successfully downloaded digits dataset to {digits_path}") # Changed print to logger.info


    # Other datasets with raw GitHub URLs
    datasets = {
        # '3gaussians_std0.6': 'https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/3gaussians-std0.6.csv',
        # '3gaussians_std0.9': 'https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/3gaussians-std0.9.csv',
        # 'circles': 'https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/circles.csv',
        # 'moons': 'https://raw.githubusercontent.com/mohammedkhalilia/birzeit/master/comp9318/data/moons.csv'

        # 'tsne_scores : 'https://raw.reneshbedre.github.io/assets/posts/tsne/tsne_scores.csv'

        # 'driver-data': 'https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/blob/master/driver-data.csv'
        # 'vehicle': 'https://raw.githubusercontent.com/milaan9/Clustering-Datasets/blob/master/01.%20UCI/vehicle.csv'
    }

    for name, url in datasets.items():
        try:
            # Use raw GitHub URL
            raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            dataset_path = os.path.join(data_dir, f'{name}.csv')
            df = pd.read_csv(raw_url)
            df.to_csv(dataset_path, index=False)
            logger.info(f"Successfully downloaded {name} dataset to {dataset_path}") # Changed print to logger.info
        except Exception as e:
            logger.error(f"Failed to download {name}: {str(e)}") # Changed print to logger.error
            logger.error(f"URL used: {raw_url}") # Changed print to logger.error


# if __name__ == '__main__':
#     # You would now call it like this if running independently and want logging
#     # from src.utils.logger import setup_logging
#     # logger_test = setup_logging(log_file_name="download_test.log")
#     # download_datasets(logger_test)
#     pass # Keep commented out as per your original code

# Ruwa' without LLM
# def load_dataset(name):
#     """Load a dataset from the data directory"""
#     path = f'../data/{name}.csv'
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset {name} not found at {path}")
#
#     return pd.read_csv(path)
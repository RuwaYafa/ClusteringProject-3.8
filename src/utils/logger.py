# E:\ClusteringProject\src\utils\logger.py
import logging
import os
from datetime import datetime

def setup_logging(log_file_name="clustering_results-3.8v.log"):
    """
    Sets up logging to a file and the console.
    """
    # Create logs directory if it doesn't exist
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file_name)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the logging level

    # Clear existing handlers to avoid duplicate messages if called multiple times
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"--- Clustering Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    logger.info(f"Results will be saved to: {os.path.abspath(log_path)}")

    return logger
# Import required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.download_datasets import download_datasets
from src.utils.spherical_rfa import spherical_rfa

from src.utils.dbscan import dbscan
from src.utils.k_mean import kmeans, predict
from src.utils.visualization import save_visualizations
from src.utils.dbscan import dbscan_accuracy
from src.utils.logger import setup_logging # ADDED: Import setup_logging

# pip show scikit - learn Version: 1.6.1

if __name__ == '__main__':
    # Setup logging first
    logger = setup_logging() # Initialize the logger

    # Input
    # k-mean
    # datasets = download_datasets()
    k_default = 3 # Default Cluster number for datasets if not specified
    max_iter=100
    epsilon=1e-5#1e-5 #k-mean
    init_method= 'farthest' 
    # init_method= 'Random' 

    feature_names = "None"
    # DBSCAN
    eps = None #DBSCAN
    min_samples = None
    # eps = 0.6 #DBSCAN
    # min_samples = 4
    use_pca = False

    # List of datasets and their cluster numbers
    datasets = [
        ('iris', 3),
        ('3gaussians_std0.6', 3),
        ('3gaussians_std0.9', 3),
        ('circles', 2),
        ('moons', 2),
        ('digits', 10),
        ('complex9', 9),
        ('vehicle', 3) #10 #3-better than 4!!! categorical label
    ]

    # List of datasets and their cluster numbers
    """
     Data-out -high D   
    """
    # datasets = [
    #     ('digits', 10),
    #     # ('tsne_scores', 12),// not included
    #     ('complex9', 9),
    #     # ('driver-data', 55),// not included
    #     ('vehicle', 3), #10 #3-better than 4!!!
    #     # ('winequality-red', 6)// not included
    # ]

    # Main Process
    df_DS = {}

    for name, k in datasets: # Use k_val to avoid conflict with global k
        # dataset to df
        logger.info(f"\n@---- ||||||||||||| {name} Dataset ||||||||||||| ----@\n")

        """
        Data_out -high D   
        """
        path = f'data/{name}.csv'
        # path = f'../data_out/{name}.csv'

        # print('================ path ====================')
        # print(path)
        df = pd.read_csv(path)
        X = df.values[:, :-1]  # Features ( last column is label for all datasets)
        y = df.values[:, -1] if df.shape[1] > 1 else None  # Labels if exist

        df_DS[name] = {}
        df_DS[name]['df'] = df
        df_DS[name]['X'] = X
        df_DS[name]['y'] = y
        df_DS[name]['X_scaled'] = StandardScaler().fit_transform(df_DS[name]['X'])

        """
        Saved in DS-information.log
        """
        # logger.info(f"\n\n\n\n@----{name} Dataset information:"
        #             f"\n@---- shape: {df_DS[name]['X_scaled'].shape}."
        #             f"\n@---- {len(np.unique(df_DS[name]['y']))} Classes: {np.unique(df_DS[name]['y'])}."
        #             f"\n@---- Features: {len(df.columns)-1}.\n"
        #             f"----------------------------------------------------------------"
        #             f"\n@---- Description: \n{df.describe()}.\n")

        if name == 'iris':
            feature_names = df.columns
            # logger.info(f"Features: {feature_names.tolist()}") # Changed print to logger.info, .tolist() for better log output
        else:
            logger.info("Features: 2 without labels")

        from sklearn.preprocessing import LabelEncoder
        if name == 'vehicle':
            le = LabelEncoder()
            df_DS[name]['y'] = le.fit_transform(df_DS[name]['y'])




        # logger.info(f"\n@---- ||||||||||||| K-Means algorithm Dataset ||||||||||||| ----@\n")

        # # __________K-Means algorithm__________
        # final_centroids, final_labels, final_loss = kmeans(
        #     df_DS[name]['X_scaled'],
        #     k,
        #     max_iter,
        #     epsilon,
        #     init_method,
        #     name,
        #     df_DS[name]['y'], #for prediction
        #     logger=logger
        # )
        
        # # Predict labels using final centroids-----------------------
        # predicted_labels = predict(df_DS[name]['X_scaled'], final_centroids)
        
        # # If true labels exist (e.g., Iris), align and calculate accuracy
        # if df_DS[name]['y'] is not None:
        #     from sklearn.metrics import accuracy_score, confusion_matrix
        #     from scipy.optimize import linear_sum_assignment
        
        #     true = df_DS[name]['y'].astype(int)
        #     pred = predicted_labels
        
        #     # Hungarian alignment
        #     cm = confusion_matrix(true, pred)
        #     row_ind, col_ind = linear_sum_assignment(-cm)
        #     aligned_pred = np.zeros_like(pred)
        #     for i, j in zip(col_ind, row_ind):
        #         aligned_pred[pred == i] = j
        
        #     acc = accuracy_score(true, aligned_pred)
        #     logger.info(f"{name}-K_means Prediction Accuracy (Hungarian Matched): {acc}")
        
        
        # print(f"Converged in {len(np.unique(final_labels))} clusters")
        # print("Final loss:", final_loss)
        # print("Prediction Accuracy (Hungarian Matched):", acc)
        
        # # Generate visualizations for k-mean
        # save_visualizations(
        #     df_DS[name]['X_scaled'],
        #     final_labels,
        #     final_centroids,
        #     feature_names, # iris
        #     name,
        #     k,
        #     epsilon,
        #     max_iter
        # )
        
        # # # Parameter sensitivity analysis
        # # print("\n=== Parameter Sensitivity ===")
        # # for ks in [2, 3, 4, 5]: #, 6, 7, 8, 9, 10, 11, 12]:
        # #     _ = kmeans(
        # #             df_DS[name]['X_scaled'],
        # #             ks,
        # #             max_iter,
        # #             epsilon,
        # #             init_method,
        # #             name
        # #         )
        
        


        logger.info(f"\n@---- ||||||||||||| DBSCAN algorithm ||||||||||||| ----@\n")

        # __________DBSCAN Algorithm with or without pca__________
        labels_nd = dbscan(
            df_DS[name]['X_scaled'],
            df_DS[name]['y'],
            name,
            eps,
            min_samples,
            use_pca=False, logger=logger
        )

        # Compare with pca
        labels_2d = dbscan(
            df_DS[name]['X_scaled'],
            df_DS[name]['y'],
            name,
            eps,
            min_samples,
            use_pca=True,
            logger=logger
        )

        # # # Parameter sensitivity analysis
        # # print("\n=== Parameter Sensitivity ===")
        # # for eps in [0.3, 0.5, 0.7, 0.8]:
        # #     _ = dbscan(
        # #         df_DS[name]['X_scaled'],
        # #         df_DS[name]['y'],
        # #         name,
        # #         eps,
        # #         min_samples,
        # #         use_pca=False
        # #     )
        #
        acc_labels_nd = dbscan_accuracy(df_DS[name]['y'],labels_nd)
        logger.info(f"{name}-DBSCAN-nd Prediction Accuracy (Hungarian Matched): {acc_labels_nd}")
        print(acc_labels_nd)
        acc_labels_2d = dbscan_accuracy(df_DS[name]['y'],labels_2d)
        logger.info(f"{name}-DBSCAN-2d Prediction Accuracy (Hungarian Matched): {acc_labels_2d}")
        print(acc_labels_2d)


        
        # logger.info(f"\n@---- ||||||||||||| spherical algorithm Dataset ||||||||||||| ----@\n")
        # # __________spherical Algorithm__________
        # # Pass the logger instance
        # final_labels, final_centroids = spherical_rfa(
        #     df_DS[name]['X_scaled'],
        #     df_DS[name]['y'],
        #     name,
        #     k, # Use k_val from the dataset tuple or custom
        #     # logger=logger # ADDED: Pass the logger
        # )
        
        # # Predict labels using final centroids-----------------------
        # predicted_labels = predict(df_DS[name]['X_scaled'], final_centroids)
        
        # # If true labels exist (e.g., Iris), align and calculate accuracy
        # if df_DS[name]['y'] is not None:
        #     from sklearn.metrics import accuracy_score, confusion_matrix
        #     from scipy.optimize import linear_sum_assignment
        
        #     true = df_DS[name]['y'].astype(int)
        #     pred = predicted_labels
        
        #     # Hungarian alignment
        #     cm = confusion_matrix(true, pred)
        #     row_ind, col_ind = linear_sum_assignment(-cm)
        #     aligned_pred = np.zeros_like(pred)
        #     for i, j in zip(col_ind, row_ind):
        #         aligned_pred[pred == i] = j
        
        #     acc = accuracy_score(true, aligned_pred)
        #     logger.info(f"{name}-Spherical_K_means Prediction Accuracy (Hungarian Matched): {acc}")

    logger.info("\n--- All Clustering Processes Completed ---") # Final log message
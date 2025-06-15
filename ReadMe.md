## Machine learning Project- Kmeans - Dr. Mohammed Khalilia
'''
Ruwa F. AbuHweidi @ 2025
rabuhweidi@birzeit.edu
'''

![img.png](img.png)

### Code structure:
"""
.
├── .venv/
├── data/
│   ├── 3gaussians_std0.6.csv
│   ├── 3gaussians_std0.9.csv
│   ├── circles.csv
│   ├── iris.csv
│   └── moons.csv
├── dbscan_results/
├── kmean-results/
├── logs/
│   └── clustering_results-3.8.log
├── spherical-results/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dbscan.py
│   │   ├── download_datasets.py
│   │   ├── k_mean.py
│   │   ├── logger.py
│   │   ├── spherical_rfa.py
│   │   └── visualization.py
│   ├── __init__.py
├── main.py
├── README.md
├── requirement3.8.txt
└── requirement3.12.txt
"""
### How run the code:
In cli:
python main.py
args ... 


### Step 1: Understand the Project Requirements
Main Task: Implement k-means clustering from scratch in Python

│   │   ├── k_mean.py
Base code: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

Additional Tasks:
Compare with DBSCAN and Spherical Clustering (using existing implementations)
Apply to provided datasets

│   │   ├── spherical_rfa.py
Base code inspiring from: https://github.com/jasonlaska/spherecluster
package: https://pypi.org/project/spherecluster/0.1.7/#spherecluster-0.1.7-py3-none-any.whl

│   │   ├── dbscan.py
base code: https://stackoverflow.com/questions/61233304/clustering-with-dbscan-how-to-train-a-model-if-you-dont-set-the-number-of-clust
Visualize results (including dimensionality reduction for high-dimension data)
Write a comprehensive report in LaTeX

│   │   └── visualization.py

### Step 2: Set Up Your Development Environment
Install required Python packages: 
All clustering algorithm work on python 3.8
All clustering algorithm work on python 3.12 except spherical clustering
      
### Step 3: Implement k-Means from Scratch
Here's how to approach the implementation:

### Pseudocode for k-Means:
1. Initialize k centroids randomly from data points
2. Repeat until convergence:
   a. Assign each point to nearest centroid (Euclidean distance)
   b. Update centroids as mean of assigned points
   c. Check if centroids changed significantly

### Python Implementation Steps:

The project implemented in python 3.12.
you can run k_mean and DBSCAN except Spherical clustering.
It implemented in python 3.8, with slightly change in the spherical library 
https://github.com/jasonlaska/spherecluster.git
https://pypi.org/project/spherecluster/0.1.7/#spherecluster-0.1.7-py3-none-any.whl

if you run the code in just comment the Call of spherical_rfa in the end of main.py file

### Hints to run spherical_rfa.py:

python --version
Python 3.8.10 
https://www.youtube.com/watch?v=AUiM1UaRCPc  
https://www.python.org/downloads/release/python-3810/

pip install spherecluster==0.1.7  
https://pypi.org/project/spherecluster/0.1.2/#description
pip debug --verbose
pip install requests

from -> https://www.cgohlke.com/
pip install setuptools-50.0.0-py3-none-any.whl  # https://pypi.org/project/setuptools/50.0.0/
pip install setuptools==50.0.0
pip install pip-20.2.4-py2.py3-none-any.whl > in download  # https://www.piwheels.org/project/pip/
pip install scikit-learn==0.22  # https://pypi.org/project/scikit-learn/0.22/#history
pip install requests

#### pip freeze
pip install -r requirement3.8.txt
pip install -r requirement3.12.txt

##### Acknoledgment 
We want to thank Professor M. Khalilia for his professional educational approach, and his kindness in Imitation Learning when he gives us free guidance, without limitation, and opens his code on GitHub: https://github.com/mohammedkhalilia. 
Also, we want to thank Large Language Models for their gentle supervision and their respect for our prompts to give the advice without changing our work, only if we clearly ask him, especially in visualization code.

RuwaYafa@2025

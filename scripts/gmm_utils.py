import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def evaluate_gmm(n_c, n_p, bbj_data_values, eigen_weights):
    """
    Run GMM clustering for a specific combination of PCs and components.
    
    Parameters:
    - n_c: Number of components (clusters)
    - n_p: Number of PCs to use
    - bbj_data_values: Numpy array of SHAPE (n_samples, n_total_pcs). 
                      Must clear columns before passing or index inside.
                      To minimize serialization, pass only the array.
    - eigen_weights: Weights for all PCs.
    """
    try:
        # Slice the data and weights for the current number of PCs
        # Assuming bbj_data_values columns match eigen_weights length or more
        
        current_data = bbj_data_values[:, :n_p] * eigen_weights[:n_p]
        
        # Fit GMM
        # covariance_type='full' allows clusters to have different shapes/orientations
        # n_init=3 runs k-means initialization 3 times and takes best to avoid local optimals
        gmm = GaussianMixture(n_components=n_c, covariance_type='full', random_state=42, n_init=3, init_params='kmeans')
        labels = gmm.fit_predict(current_data)
        
        # Calculate Metrics
        # BIC: Lower is better. Handles model complexity penalty.
        bic = gmm.bic(current_data)
        
        # Silhouette: Higher is better.
        if len(current_data) > 10000:
            idx = np.random.choice(len(current_data), 10000, replace=False)
            sil_score = silhouette_score(current_data[idx], labels[idx])
        else:
            sil_score = silhouette_score(current_data, labels)
            
        return {
            'n_pcs': n_p,
            'n_components': n_c,
            'bic': bic,
            'silhouette': sil_score
        }
    except Exception as e:
        return {
            'n_pcs': n_p,
            'n_components': n_c,
            'bic': np.inf,
            'silhouette': -1.0,
            'error': str(e)
        }

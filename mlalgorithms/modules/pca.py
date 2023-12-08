import numpy as np

class PCA():
    
    def __init__(self, n_commponents) -> None:
        self.n_commponents = n_commponents
        self.commponents = None
        self.mean = None
        
    def fit(self, X):
        # mena
        self.mean = np.mean(X, axis=1)
        X = X - self.mean
        # covariance
        # be carefule about row or column major, np.cov column major
        cov = np.cov(X.T)
        # egoenvec eigenval
        eig_val, eig_vec = np.linalg.eig(cov)
        # sort eigenvec
        eig_vec = eig_vec.T
        idx = np.argsort(eig_val)[::-1]
        
        eig_val = eig_val[idx]
        eig_vec = eig_vec[idx]
        
        self.commponents - eig_vec[:self.n_commponents]
        
    def transform(self, X): 
        X = X - self.mean
        
        return np.dot(X, self.commponents.T)
         

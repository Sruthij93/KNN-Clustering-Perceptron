from random import random
import numpy as np
# from matplotlib import plot
from scipy.spatial import distance

def find_clusters(X, mu, K):
    """function to calculate the clusters for given cluster centres"""
    cluster= [[] for _ in range(K)] 
    for i in range(X.shape[0]):
        d=[]
        for j in mu:
            d.append(distance.euclidean(np.atleast_1d(X[i]),np.atleast_1d(j)))
        d_min = min(d)
        d_min_idx= d.index(d_min)
        # print("d_min_idx==", d_min_idx)
        cluster[d_min_idx].append(X[i])
        
    return(cluster)


def K_Means(X,K,mu):
    # checking if mu is empty to generate random cluster centres
    if(mu.size == 0):
        mu = X[np.random.choice(X.shape[0], K, replace=True)] 
        if(mu.ndim ==1):
            mu = mu.reshape(-1, 1)
    
    X = X.astype(np.int64)

    mu = np.atleast_1d(mu.squeeze())
    
    # print("mu = ", mu)

    mu_new = []
    new_cluster = []
    cluster = []
    
    # calculating clusters and update cluster centres based on new clusters
    update = True

    while update:
        cluster = find_clusters(X, mu, K)
        mu_new = []
        for i in range(len(cluster)):
            
            if(np.isnan(cluster[i]).all()):
                means = 0
            else:
                means = np.mean(cluster[i], axis=0)
                means = np.round(means, 3)
                means = means.tolist()
            mu_new.append(means)
        # print("new mu = ", mu_new)
        new_cluster = find_clusters(X, mu_new, K)
        # print("new cluster = ", new_cluster)
    
        if np.array_equal(mu_new, mu):
            update = False
        else:
            mu = mu_new
    # print(cluster)
    return mu    
        
        

    
    




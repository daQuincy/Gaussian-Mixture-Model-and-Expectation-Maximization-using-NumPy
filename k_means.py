# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:41:03 2019

@author: YQ
"""

import numpy as np

class KMeans:
    def __init__(self, n_centroids, n_iters):
        self.n_centroids = n_centroids
        self.n_iters = n_iters
        
    def get_distance(self, p1, p2):
        d = np.linalg.norm(p1-p2)
        
        return d
    
    def initialize_centroids(self):
        c = []
        for i in range(self.n_centroids):
            idx = np.random.choice(len(self.X))
            c.append(self.X[idx, :])
            
        return np.array(c)
    
    def assign_centroids(self, centroids):
        all_centroids = []
        for i in range(len(self.X)):
            smallest = np.infty
            assigned_centroid = None
            for ii in range(self.n_centroids):
                d = self.get_distance(self.X[i,:], centroids[ii,:])
                if d < smallest:
                    assigned_centroid = ii
                    smallest = d
            all_centroids.append(assigned_centroid)
         
        all_centroids = np.array(all_centroids)
        all_assigned = []
        for i in range(self.n_centroids):
            temp = self.X[all_centroids==i]
            all_assigned.append(temp)
            
        return all_assigned
    
    def update_centroids(self, all_assigned):
        updated_centroid = []
        for i in all_assigned:
            temp_y, temp_x = np.mean(i[:,1]), np.mean(i[:,0])
            updated_centroid.append((temp_x, temp_y))
            
        return np.array(updated_centroid)
    
    def fit(self, X):
        self.X = X
        centroids = self.initialize_centroids()
        
        for i in range(self.n_iters):
            all_assigned = self.assign_centroids(centroids)
            updated_centroid = self.update_centroids(all_assigned)
            centroids = updated_centroid
            
        self.centroids = centroids
        self.assigned = all_assigned
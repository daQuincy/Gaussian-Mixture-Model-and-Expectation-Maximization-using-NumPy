# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:23:54 2019

@author: YQ
"""

# TODO: implement multivariate normal from scratch without using scipy

from scipy.stats import multivariate_normal
from scipy.special import softmax
from sklearn.datasets import make_spd_matrix
import numpy as np

class GMM:
    def __init__(self, C=3, rtol=1e-3, max_iter=100, restarts=10):
        self.C = C                   # number of mixtures 
        self.rtol = rtol             # minimum change during optimization
        self.max_iter = max_iter     # maximum training iterations
        self.restarts = restarts     # number of random restarts
        
    def multi_normal(self, X, mean, cov):
        """
        Probability Density Function of Multivariate Normal distribution
        X: (N x d), data points
        mean
        """
        
        mu = np.expand_dims(mean, -1)
        x = np.expand_dims(X, -1)
        x_mu = x - mu
        
        normalizer = 1 / np.sqrt(np.linalg.det(2 * np.pi * cov))
        term = np.exp(-0.5 * np.matmul(x_mu.T, np.matmul(np.linalg.inv(cov) , x_mu)))
        pdf = normalizer * term
        
        return pdf
        
    def E_step(self, X, pi, mu, sigma):
        """
        Performs E-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        pi: (C), mixture component weights 
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices
        
        Returns:
        gamma: (N x C), probabilities of clusters for objects
        """
        N = X.shape[0] # number of objects
        C = pi.shape[0] # number of clusters
        gamma = np.zeros((N, C)) # distribution q(T)
        
        for i in range(N):
          #tmp = [pi[cc] * multivariate_normal(mean=mu[cc], cov=sigma[cc]).pdf(X[i]) for cc in range(C)]
          tmp = [pi[cc] * self.multi_normal(X=X[i], mean=mu[cc], cov=sigma[cc]) for cc in range(C)]
          denominator = np.sum(tmp)
          
          for idx, ii in enumerate(tmp):
            tmp_gamma = ii / denominator
            gamma[i, idx] = tmp_gamma
        
        return gamma
    
    def M_step(self, X, gamma):
        """
        Performs M-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)  
        
        Returns:
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        N = X.shape[0] # number of objects
        C = gamma.shape[1] # number of clusters
        
        mu, sigma = [], []
    
        N_k = np.sum(gamma, 0)
        
        pi = N_k / N
        
        for c in range(C):
          mu_k = []
          for n in range(N):
            x = np.expand_dims(X[n], -1)
            mu_tmp = gamma[n, c] * x
            mu_k.append(mu_tmp)
            
          
          mu_k = np.squeeze(np.stack(mu_k), -1)
          mu_k = np.sum(mu_k, 0)
          mu.append((1/N_k[c]) * mu_k)
          
        mu = np.stack(mu, 0)
        
        for c in range(C):
          sigma_k = []
          mu_tmp = np.expand_dims(mu[c], -1)
          for n in range(N):
            x = np.expand_dims(X[n], -1)
            
            sigma_tmp = gamma[n, c] * np.matmul((x-mu_tmp), (x-mu_tmp).T)
            sigma_k.append(sigma_tmp)
            
          sigma_k = np.stack(sigma_k)
          sigma_k = np.sum(sigma_k, 0)
          sigma.append((1/N_k[c]) * sigma_k)
        
        sigma = np.stack(sigma, 0)
        
        return pi, mu, sigma
    
    def vlb(self, X, pi, mu, sigma, gamma):
        """
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)  
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        
        Returns value of variational lower bound
        """
        N = X.shape[0] # number of objects
        C = gamma.shape[1] # number of clusters
    
        term1 = []
        term2 = []
        for n in range(N):
          for c in range(C):
            term1_tmp = gamma[n, c] * (np.log(pi[c]+1e-20) + multivariate_normal(mu[c], sigma[c]).logpdf(X[n]))
            term2_tmp = gamma[n, c] * np.log(gamma[n, c]+1e-20)
            
            term1.append(term1_tmp)
            term2.append(term2_tmp)
                       
        term1 = np.sum(term1)
        term2 = np.sum(term2)
        
        loss = term1 - term2
    
        return loss
    
    def fit(self, X):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.
        
        X: (N, d), data points
        C: int, number of clusters
        '''
        d = X.shape[1] # dimension of each object
        best_loss = 9999999.0
        best_pi = None
        best_mu = None
        best_sigma = None
        
    
        for r in range(self.restarts):
            loss0 = 999.0
            pi0 = softmax(np.random.normal(size=(self.C,)))
            mu0 = np.random.normal(size=(self.C, d))
            sigma0 = np.stack([make_spd_matrix(d) for _ in range(self.C)], 0)
            try:
                for i in range(self.max_iter):
                  gamma = self.E_step(X, pi0, mu0, sigma0)
                  pi, mu, sigma = self.M_step(X, gamma)
                  loss = self.vlb(X, pi, mu, sigma, gamma)
    
                  # best parameters
                  if abs(loss) < abs(best_loss):
                    best_loss = loss
                    best_pi = pi
                    best_mu = mu
                    best_sigma = sigma
    
                  # early stopping
                  threshold = abs((loss-loss0)/loss0)
                  if threshold <= self.rtol:
                    print("Early stopping")
                    break
                
                  if np.isnan(loss):
                      print("NaN value occured")
                      break
    
                  loss0 = loss
                  pi0 = pi
                  mu0 = mu
                  sigma0 = sigma
    
                  print("Restart: {}   Iteration: {}   Loss: {:.4f}".format(r+1, i+1, loss))
    
            except Exception as e: #np.linalg.LinAlgError:
                #print("Singular matrix: components collapsed")
                print(e)
                pass
    
        self.best_loss = best_loss 
        self.best_pi = best_pi 
        self.best_mu = best_mu 
        self.best_sigma = best_sigma
        self.gamma = self.E_step(X, best_pi, best_mu, best_sigma)
        
        
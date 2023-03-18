#----------------------------------------------#
#-------| Written By: Muhammad Shaffay |-------#
#----------------------------------------------#

# A K Nearest Neighbor algorithm ...
import numpy as np
import pandas as pd

class KNearestNeighbor:
    ''' Implements the KNearest Neigbours For Classification... '''
    def __init__(self, k, scalefeatures=False):  
        
        self.k = k
        self.scalefeatures = scalefeatures
        
        self.X_train = []    
        self.Y_train = []     
    
    def compute_distances_two_loops(self, X):

        num_test = X.shape[0] # no. of test examples
        num_train = self.X_train.shape[0] # no. of train examples
        
        dists = np.zeros((num_test, num_train)) # a matrix of dimension = (num_test X num_train)

        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(((self.X_train[j] - X[i]) ** 2).sum())
            
        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(((self.X_train - X[i]) ** 2) , axis=1))

        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        
        dists = np.zeros((num_test, num_train)) 
        
        dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))

        return dists

    def scale_features(self,X):
        
        df = pd.DataFrame(X)
        
        mini , maxi = [] , []
        for i in df.columns:
            mini.append(min(df[i]))
            maxi.append(max(df[i]))
            
        # min-max scaler
        self.xmin , self.xmax = np.array(mini) , np.array(maxi)
        X = (X - self.xmin) / (self.xmax - self.xmin)

        return X
        
    def train(self, X, Y):
        
        nexamples , nfeatures = X.shape
        
        # storing values for further computation
        if self.scalefeatures:
            X=self.scale_features(X)
            
        self.X_train = X    
        self.Y_train = Y 
        
    
    def predict(self, X, methodtype='noloops'):
        
        num_test = X.shape[0]
        
        if self.scalefeatures:
            X=(X-self.xmin)/(self.xmax-self.xmin)
        
        y_pred = np.zeros(num_test, dtype = self.Y_train.dtype)
        
        # defining a function variable so that you will only need to call compute_distance...
        if methodtype == 'noloops':
            compute_distance = self.compute_distances_no_loops
        elif methodtype == 'oneloop':
            compute_distance = self.compute_distances_one_loop
        else:
            compute_distance = self.compute_distances_two_loops

        dists=compute_distance(X)
        for i , value in enumerate(dists): # picking each example
            
            sorted_array = np.argsort(value) # sorting

            nearest_neighbors = [] 
            for j in range(self.k): # picking neighbour one by one
                
                index = sorted_array[j]
                nearest_neighbors.append(self.Y_train[index])

            y_pred[i] = max(set(nearest_neighbors), key = nearest_neighbors.count) # checking most occuring value
            
        return y_pred

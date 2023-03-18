#----------------------------------------------#
#-------| Written By: Muhammad Shaffay |-------#
#----------------------------------------------#

# A Gaussian Naive Bayes Classifier ...

import numpy as np

def univariate_gaussian(constant , std, mean, x):
    
    return (constant ** (-0.5 * (((x - mean) / std) ** 2))) 

class GaussianBayes:
    
    def __init__(self):
        ''' Implements the Gaussian Naive Bayes For Classification... '''
        self.k = []
        self.labels = []
        
        self.mean = []
        self.std = []
        self.constant = []
        self.features = 0
        
    
    def train(self, X, Y):
        
        # Calculate No. Of Classes To Be Predicted
        self.k = len(np.unique(Y))
        print("Training a " + str(self.k) + "-class Gaussian Naive Bayes Classifier with " + str(len(X[0]))  + " features" )
        
        # Extracting Unique Values
        self.labels = np.unique(Y)
            
        # Separating Data On The Basis Of Labels
        grouped_data = []
        for i , label in enumerate(self.labels):
            temp_array = []
            for j , value in enumerate(X):
                if label == Y[j]:
                    temp_array.append(value)
            grouped_data.append(temp_array)
                
        # Calculating Total Features
        self.features = len(X[0])
        
        # Calculate Mean Of each Feature For Every Class
        means = []
        for i in grouped_data:
            means.append(np.mean(i, axis = 0))
        self.mean = means

        # Calculate Standard Deviations For Every Class
        standarddev = []
        for i in grouped_data:
            standarddev.append(np.std(i, axis = 0))
        self.std = standarddev
        
        # Calculating Constants
        constant_array = []
        for i in range(self.k):
            temp_array = []
            for j in range(self.features):
                temp = (1 / (self.std[i][j] * ((2 * np.pi) ** 0.5))) * 2.718
                temp_array.append(temp)
            constant_array.append(temp_array)
        self.constant = constant_array  
        
        
    def test(self, X):
        
        #testing with univariate gaussian
        
        assigned_classes = []
        probabilities = []
        
        probability = []
        for i , datapoint in enumerate(X):
            array = []
            for k in range(self.k):
                p = 1
                for j , value in enumerate(datapoint):        
                    p *= (univariate_gaussian(self.constant[k][j], self.std[k][j], self.mean[k][j], value))
                    
                array.append(p)
            indexes = np.argsort(array)
            index = indexes[-1]
            
            assigned_classes.append(self.labels[index])
            probabilities.append(array[index])
  
        return assigned_classes , probabilities
    
    def predict(self, X):
        return self.test(X)[0]    

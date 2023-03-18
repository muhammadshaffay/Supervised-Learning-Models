#----------------------------------------------#
#-------| Written By: Muhammad Shaffay |-------#
#----------------------------------------------#

# A Gaussian Bayes Classifier ...

import numpy as np

def multivariate_gaussian(determinant , inverse, k, x, means):
    
        value1 = 1 / ( ( ( (2 * 3.13) ** k ) * determinant ) ** 0.5)

        value2 = (-0.5 * ( (x - means).T ) )
        value2.reshape(len(means),1)

        value3 = np.dot(inverse , (x - means))

        result = value1 * (2.718 ** np.dot(value2 , value3))

        return result

class GaussianBayes:
    
    ''' Implements Gaussian Bayes Classifier... '''    
    def __init__(self):

        __k = 0 # No. of classes to be predicted
        __labels = [] # Classes names
        
        __means_per_class = []
        __covariance_per_class = []
        __inverse_per_class = []
        __determinants_per_class = []
        __probability_per_class = []

    
    def train(self, X, Y):
        
        # Calculate No. Of Classes To Be Predicted
        self.k = len(np.unique(Y))
        print("Training a " + str(self.k) + "-class Gaussian Bayes Classifier with " + str(len(X[0]))  + " features" )
        
        # Extracting Unique Values
        self.labels = np.unique(Y)

        # Calculating Probability Per Class
        prob_list = []
        for i , label in enumerate(self.labels):
            count = 0
            for j , value in enumerate(X):
                if label == Y[j]:
                    count += 1
            prob_list.append(count/len(X))
        self.probability_per_class = prob_list
        
        # Separating Data On The Basis Of Labels
        grouped_data = []
        for i , label in enumerate(self.labels):
            temp_array = []
            for j , value in enumerate(X):
                if label == Y[j]:
                    temp_array.append(value)
            grouped_data.append(temp_array)
        
        # Calculate Mean Of each Feature For Every Class
        means = []
        for i in grouped_data:
            means.append(np.mean(i, axis = 0))
        self.means_per_class = means

        # Calculate Covariance Matrix For Every Class
        covariance = []
        for i in grouped_data:
            covariance.append(np.cov((np.array(i)).T))
        self.covariance_per_class = covariance

        # Calculate Inverse Matrix For Every Class
        inverse = []
        for i in self.covariance_per_class:
            inverse.append(np.linalg.inv(i))
        self.inverse_per_class = inverse

        # Calculate Deteminants For Every Class
        determinants = []
        for i in self.covariance_per_class:
            determinants.append(np.linalg.det(i))
        self.determinants_per_class = determinants        

    def test(self, X):

        assigned_classes = []
        probabilities = []
        
        # Evaluating
        for i , datapoint in enumerate(X):
            probability_per_class = []
            for j in range(self.k): 
                p = multivariate_gaussian(self.determinants_per_class[j], self.inverse_per_class[j], self.k, datapoint, self.means_per_class[j])
                probability_per_class.append(p * self.probability_per_class[j])
             
            indexes = np.argsort(probability_per_class)
            index = indexes[-1]
            
            assigned_classes.append(self.labels[index])
            probabilities.append(probability_per_class[index])
            
        return assigned_classes , probabilities 
        
    def predict(self, X):
        return self.test(X)[0]  

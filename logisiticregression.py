#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *

#Note: Here the bias term is considered as the last added feature 

class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''

    def __init__(self, lembda):        
        """
            lembda= Regularization parameter...            
        """
        #Classifier.__init__(self, lembda)                
        
        self.theta = []
        self.lembda = lembda

    def sigmoid(self, z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """
        predicted = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(z)
        return predicted      
    
    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        z = np.dot(X, theta)
        predicted = self.sigmoid(z)

        return predicted
        
    def cost_function(self, X, Y, theta):
        '''.flatten()
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
        predicted = self.hypothesis(X, theta)
        summation = np.sum((Y * np.log(predicted)) + ((1 - Y) * np.log(1 - predicted)))
        cost = (-1/len(Y)) * summation
        regularization =  (self.lembda / (2 * len(Y))) * np.sum(theta ** 2) 

        return cost + regularization

    def derivative_cost_function(self, X, Y, theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        predicted = self.hypothesis(X, theta)
        gradient = (np.dot(X.T,  predicted - Y))  / len(Y)
        regularization = (self.lembda/len(Y)) * theta

        return gradient + regularization

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''

        self.theta = optimizer.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)
    
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X    
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        predicted = self.hypothesis(X, self.theta)
        predicted = np.where(predicted >= 0.5, 1, 0)

        return predicted
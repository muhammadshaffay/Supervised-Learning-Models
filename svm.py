# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

from classifier import *


# Note: Here the bias term is considered as the last added feature
class SVM(Classifier):
    ''' Implements the SVM For Classification... '''

    def __init__(self, lembda=0.001):
        """
            lembda= Regularization parameter...
        """
        #Classifier.__init__(self, lembda)
        self.theta = []
        self.lembda = lembda

    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''

        return np.dot(X , theta)     

    def cost_function(self, X, Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector

            Return:
                Returns the cost of hypothesis with input parameters
        '''
        predicted = self.hypothesis(X, theta)
        loss = 1 - (Y * predicted)
        loss[loss < 0] = 0
        loss = np.sum(loss)

        regularization = 1/(2 * len(Y)) * ((self.lembda / 2) * np.sum(theta ** 2))
        return loss * regularization

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
        regularization = []
        for i in range(X.shape[1]):
            loss = 0
            for j in range(X.shape[0]):
                error = max(0, 1 - (Y[j] * predicted[j]))

                if error != 0:
                    loss += -(Y[j] * X[j][i])

            regularization.append(loss + (self.lembda * theta[i]))
        
        return np.array(regularization).reshape(-1,1)

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

        nexamples, nfeatures = X.shape
        self.theta = optimizer.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)

        return
        

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

        return np.sign(predicted)
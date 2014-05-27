"""
 J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

%

%  


% 
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


"""
#function to computeCost of linear regression 
def computeCost(X,y,theta):
    #importing numpy for matrix manipulation
    import numpy as np
    #initialization of some useful parameters
    m=len(y) #number of training examples
    #predictions 
    predictions=np.dot(X,theta)
    predictionsMinusy=(predictions-y)
    sqrErrors=np.square(predictionsMinusy)#squared errors 

    #linear regression cost of a single variable 
    J=(1./(m*2)) *np.sum(sqrErrors)
    
    #Returning the cost after computation
    return J


"""
===============function to compute the cost of using theta========================
J= computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y






"""
#define a function computeCostMulti that takes 3 argument X,y and theta and returns the cost J
def computeCostMulti(X,y,theta):
    import numpy as np
    #predictions 
    predictions=np.dot(X,theta)
    #number of training examples
    m=len(y)
    #getting the errors
    predictionsminusy=predictions-y
    #calculating the cost
    predictionsminusyT=np.transpose(predictionsminusy)

    P_YT=np.dot(predictionsminusyT,predictionsminusy)
    J=(1./(2.*m)) *P_YT
    #return the calculated cost
    return J

"""
====================PYTHON IMPLEMENTATION========================
[ mu, sigma,X_norm]= featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions from numpy useful.
%  



"""
#importing some useful modules for matrix manipulation
import numpy as np
def featureNormalize(X):
    #change the numpy array to a matrix to enable manipulations using linear algebra
    X=np.matrix(X)
    #get the mean of each column of the X input that is mean of each feature
    mu=np.mean(X,0)
    #getting the standard deviation of each feature
    sigma=np.std(X,0)
    #creating a matrix of each with one column of ones to enable matrisize the mean and standard deviation
    matrisize=(np.ones((len(X),1),dtype=float))
    #making the mean and standard deviation a matrix of length of X by number of features
    matrixmu=np.matrix(matrisize*mu)
    matrixsigma=np.matrix(matrisize*sigma)
    #computing for normalized features
    X_norm=(X-matrixmu)/matrixsigma
    #unlike matlab we cant return two features in python it returns a tuple where we can separate it after return
    return [mu,sigma,X_norm]


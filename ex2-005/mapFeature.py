"""
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size




"""
degree = 6;
import numpy as np

def mapFeature(X1,X2):
    out = np.ones(np.size(X1[:]));
    out=out.reshape(np.size(X1[:]),1)
    out=np.matrix(out)
    for j in range (1,degree+1):
        for i in range (j+1):
            out=np.column_stack((X1 ** (i-j) * X2 ** j,out))

    return out


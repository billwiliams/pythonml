"""
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).



"""
def sigmoid(z):
    #importing useful modules i.e. numpy for linear algebra manipulations
    import numpy as np

    g=1.0/(1.0 + np.exp(-z))

    return g




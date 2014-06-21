"""
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%

% You need to return the following variables correctly




"""

#importing modules
def randInitializeWeights(L_in,L_out):
    import numpy as np

    W = np.zeros((L_out, 1 + L_in),dtype=float);

    epsilon_init = 0.12;
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init * epsilon_init;

    return W




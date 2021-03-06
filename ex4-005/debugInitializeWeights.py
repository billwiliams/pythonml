"""
function W = debugInitializeWeights(fan_out, fan_in)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%

% Set W to zeros



"""
def debugInitializeWeights(fan_out, fan_in):
    #importing some useful modules
    import numpy as np


    W = np.zeros((fan_out, 1 + fan_in),dtype=float);

    W = np.reshape(np.sin(np.r_[1:np.size(W)+1]), np.shape(W),order='F') / 10.0;
    
    return W


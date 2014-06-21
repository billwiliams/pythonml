"""
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
                             %   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
"""
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,\
                   num_labels,X,y,lamda):
    import numpy as np
    import sigmoid as sg
    import sub2ind
    import sigmoidGradient as sG
    #restructuring nn_params back to Theta1 and Theta2 python has a o based indexing unlike matlab which has 1 based indexing
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, (input_layer_size + 1)));
    Theta2 = np.reshape(nn_params[((hidden_layer_size * (input_layer_size + 1))):len(nn_params)],(num_labels, (hidden_layer_size + 1)));
    # Setup some useful variables
    m =np.size(X, 0);
             
    # You need to return the following variables correctly 
    J = 0;
    Theta1_grad = np.zeros(np.size(Theta1),dtype=float);
    Theta2_grad = np.zeros(np.size(Theta2),dtype=float);
    """
    % Part 1: Feedforward the neural network and return the cost in the
    %         variable J. After implementing Part 1, you can verify that your
    %         cost function computation is correct by verifying the cost
    %         computed in ex4.m
    %
    % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. After implementing Part 2, you can check
    %         that your implementation is correct by running checkNNGradients
    %
    %         Note: The vector y passed into the function is a vector of labels
    %               containing values from 1..K. You need to map this vector into a 
    %               binary vector of 1's and 0's to be used with the neural network
    %               cost function.
    %
    %         Hint: We recommend implementing backpropagation using a for-loop
    %               over the training examples if you are implementing it for the 
    %               first time.
    %
    % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    %
    """
    num_labels=np.size(Theta2,0)

    a1=np.r_[np.ones((1,m),dtype=float),X.conj().T]
    z2 = np.dot(Theta1 , a1);
    a2 =np.r_[np.ones((1, m),dtype=float), sg.sigmoid(z2)]; # 26 x m
    a3 = sg.sigmoid(np.dot(Theta2 , a2)); # 10 x m
    # Explode y into 10 values with Y[i] := i == y.
 
    Y = np.zeros((num_labels, m),dtype=float).flatten(1);
    Y[np.asarray(sub2ind.sub2ind(np.shape(Y),y.T,(np.arange(m).reshape(1,m,order='F'))))-1]=1;
    Y=Y.reshape(10,m)
    
    
    J = (1.0/m) * np.sum(np.sum((-Y*np.log(a3)) -( (1 - Y) * np.log(1 - a3))));
    # Add regularized error. Drop the bias terms in the 1st columns.
    J = J + (lamda / (2*m)) * np.sum(np.sum(Theta1[:, 1:] ** 2));

    J = J + (lamda / (2*m)) * np.sum(np.sum(Theta2[:, 1:] ** 2));
    # 2. Backpropagate to get gradient information.
    
    d3 = a3 - Y; # 10 x m
    d2 = (np.dot(Theta2.conj().T , d3)) * np.r_[np.ones((1, m),dtype=float) ,sG.sigmoidGradient(z2)];#  26 x m
    # Vectorized ftw:
    Theta2_grad = (1/m) *np.dot( d3 , a2.conj().T);
    Theta1_grad = (1/m) * np.dot(d2[1:, :], a1.conj().T);
    # Add gradient regularization.
    Theta2_grad = Theta2_grad +  (lamda / m) * (np.c_[np.zeros((np.size(Theta2, 0), 1),dtype=float), Theta2[:, 1:]])
    Theta1_grad = Theta1_grad +  (lamda / m) * (np.c_[np.zeros((np.size(Theta1, 0), 1),dtype=float), Theta1[:, 1:]])
    grad=np.r_[Theta1_grad.flatten(1),Theta2_grad.flatten(1)]

    return [J,grad]

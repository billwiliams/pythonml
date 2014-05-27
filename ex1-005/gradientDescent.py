"""
theta, J_history= gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%
% Hint: While debugging, it can be useful to print out the values
%       of the cost function (computeCost) and gradient here.
%theta = theta - (alpha/m) * (X' * (X*theta-y));

% ============================================================
% Save the cost J in every iteration    
J_history[iter,0]= computeCost(X, y, theta);     



"""
def gradientDescent(X,y,theta,alpha,num_iters):
    #neccesary modules
    import numpy as np
    #importing the computeCost to calculate the cost of using theta
    import computeCost as cC
    m=len(y)#number of training examples
    J_history=np.zeros((num_iters,1),dtype=float)#initializing cost to zeros
    #loop to update theta in evry iteration
    for iter in range(num_iters):
        #update theta in every iteration
        theta=theta-(alpha/m) * np.dot(X.T , ((np.dot(X,theta))-y))
        #save the cost after every iteration
    
        J_history[iter,0]=(cC.computeCost(X,y,theta))
    #return the cost and theta values
    return [theta,J_history]


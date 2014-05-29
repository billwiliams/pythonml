"""
 [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.




"""

def costFunction(theta,X,y):
    #importing some useful useful modules
    import numpy as np
    import sigmoid as sg
    
    #setting predictions
    m=len(y)
    #calculating predictions using the sigmoid function
    predictions=sg.sigmoid(np.dot(X,theta))
    predictionsminusy=predictions-y
    #calculating the cost of using theta as per logistic regression equation
    logistic=(-y*np.array(np.log(predictions)))-((1-y)*np.array(np.log(1-predictions)))
    J=1./m * np.sum(logistic)
    #computing partial derivatives w.r.t to parameters

    grad=1./m *np.dot( (np.transpose(X) ),(predictions-y))
    #returning the values computed
    
    

    return [J,grad]

"""
COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters




"""
def costFunctionReg(theta, X, y, labda):
    import numpy as np
    import sigmoid as sg

    grad=np.zeros((1,1),dtype=float)

    #setting predictions
    m=len(y)
    #calculating predictions using the sigmoid function
    predictions=sg.sigmoid(np.dot(X,theta))
    y=y.reshape(np.size(y),1)
    predictionsminusy=predictions-y
    predictionsminusyr=predictionsminusy.reshape(1,np.size(predictionsminusy))
    #calculating the cost of using theta as per logistic regression equation
    logistic=(-y*np.array(np.log(predictions)))-((1-y)*np.array(np.log(1-predictions)))
    endr,endc=np.shape(theta)
    J=1./m * np.sum(logistic)+(labda/(2*m))*np.sum(theta[0:endc,0]**2)
    #computing partial derivatives w.r.t to parameters
    endg=np.size(grad)
    endx,end=np.shape(X)
    grad[0,0]= 1.0/m * np.sum(np.dot((X[:,0]) ,(predictionsminusyr) ));
    
    gra=1./m * (np.dot((predictionsminusy.T), (X[:,1:end]))).T + (labda/m)*theta[1:endr]
    
    grad=np.insert(grad,1,gra,0)
    


    return [J,grad]
                                                



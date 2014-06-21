"""
COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters




"""
def costFunctionReg(theta, X, y, labda):
    import numpy as np
    import sigmoid as sg

    grad=np.zeros((28,1),dtype=float)

    #setting predictions
    m=len(y)
    #calculating predictions using the sigmoid function
    predictions=sg.sigmoid(np.dot(X,theta))
    predictions=predictions.reshape(np.size(predictions),1)
    y=y.reshape(np.size(y),1)
    predictionsminusy=predictions-y
    #predictionsminusyr=predictionsminusy.reshape(np.size(predictionsminusy),1)
    #calculating the cost of using theta as per logistic regression equation
    logistic=(-y*np.array(np.log(predictions)))-((1-y)*np.array(np.log(1-predictions)))

    #getting the size of theta one could use r,c=numpy.shape(theta) where r=rows and c=columns
    endr=np.size(theta)
    #calculating the cost with regularization parameter labda
    J=1./m * np.sum(logistic)+(labda/(2*m))*np.sum(theta[0:endr]**2)
    #computing partial derivatives w.r.t to parameters
    #getting the size of gra
    endg=np.size(grad)
    #getting the number of rows and colums in X
    endx,end=np.shape(X)
    #computing the value of the first gradient
    grad[0,:]= 1.0/m * np.sum(np.dot((X[:,0]) ,(predictionsminusy) ));
    #computing the values of the other gradients
    grad[1:endg,:]=((((1./m )* (np.dot(((predictionsminusy.conj().T)), (X[:,1:end]))))).conj().T) +((labda/m)*theta[1:endr]).reshape((endr-1),1);

    #combining the gradient into one matrix of size 27 by 1
    grad=grad[:];    
    

    #returning the cost .This is mainly because returning both the cost and grad results to  BFGS failing to minimize.by returning only the cos,we cac\
    #also retrieve theta using results.x and cost using results.fun.also the jacobian is set to false in the minimization
    return J
                                                



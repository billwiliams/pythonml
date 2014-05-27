"""
===================function to perform gradietn descent to learn theta========================
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
GRADIENTDESCENTMULTI Performs gradient descent to learn theta
   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
   taking num_iters gradient steps with learning rate alpha

Instructions: Perform a single gradient step on the parameter vector
               theta. 
                




"""
#define a function gradientDescentMulti that takes four arguments X,y,theta alpha and num_iters
def gradientDescentMulti(X,y,theta,alpha,num_iters):
    #import numpy as np for linear algebra manipulations
    import numpy as np
    #import compute cost function for mutiple variables
    import computeCostMulti as cCM
    J_history=np.zeros((num_iters,1),dtype=float)
    #number of training examples
    m=len(y)
    #the transpose of X.its not done in the loop since it would be very computationally expensive to do it every time
    Xtranspose=np.transpose(X)
    #for every iteration update theta
    for iter in range(0,num_iters):
        Xtheta=np.dot(X,theta)
        thetaminusy=Xtheta-y
       
        #update theta with every iteration
        theta-=np.dot((alpha/m),np.dot(Xtranspose,thetaminusy))
        
        #theta = theta + alpha * (1/m) * np.dot((y - np.dot(X,theta)),X)
        J_history[iter,0]=cCM.computeCostMulti(X,y,theta)
    #return theta and cost after the loop has executed    
    return [theta,J_history]

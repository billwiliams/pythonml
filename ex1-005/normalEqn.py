"""

 [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ============================================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
%+++++++++++++++++++Matlab++++++++++++++++++++++++++++++++++++++++++++++++++
theta = pinv(X' * X) * X' * y;%normal equation
%in python using the pinv command from numpy.linalg helps solve the equation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% -------------------------------------------------------------


% ============================================================




"""
#define a function normaEqn that takes two arguments X and y and returns theta computed using the norma equation
def normalEqn(X,y):
    import numpy as np
    #getting the transpose of X
    Xtranspose=np.transpose(X)
    #getting the dot multiplication of X' by X
    XtransposeX=np.dot(Xtranspose,X)
    #getting the dot multiplication X' by y
    Xtransposey=np.dot(Xtranspose,y)
    #computing theta by normal equation using pinv commnad from linear algebra library of numpy
    theta=np.dot(np.linalg.pinv(XtransposeX) , Xtransposey)
    #return theta computed using the normal equation
    return theta

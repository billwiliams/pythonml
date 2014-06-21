"""
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i





"""
def oneVsAll(X,y,num_labels,lamda):
    #importing modules
    
    import numpy as np
    from scipy import optimize
    import pdb as pdb
    import costFunctionReg as cFR
    #some useful variables
    m=np.size(X,0);

    n=np.size(X,1)
    
    all_theta = np.zeros((num_labels, n + 1),dtype=float);
    #adding a column of ones to X
    X=np.c_[np.ones(m),X]

    #initial theta all zeros
    initial_theta=np.zeros((n+1,1),dtype=float)
    for c in range (1,num_labels+1):
        theta=optimize.minimize(cFR.costFunctionReg, initial_theta, args=(X,np.equal(y,c).astype(int),lamda), \
                                                                               method='CG',jac=True, options={'maxiter':50})
        all_theta[c-1,:]=theta.x.conj().T

    all_theta.flatten(1)
    return all_theta


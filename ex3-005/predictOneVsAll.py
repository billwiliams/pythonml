"""
function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 




"""
def predictOneVsAll(all_theta,X):
    #importing some useful libraries 
    import numpy as np

    import sigmoid as sg
    #defining some useful libraries 
    m=np.size(X,0)

    num_labels=np.size(all_theta,0)

    X=np.c_[np.ones(m),X]

    predict =sg.sigmoid(np.dot(X , all_theta.T));
    index_max= np.argmax(predict,axis=1);
    p = (index_max+1).reshape(m,1);
    return p

"""
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)



"""
def predict(Theta1,Theta2,X):
    #importing useful libraries 

    import numpy as np
    import sigmoid as sg
    
    m=np.size(X,0)
    

    a1 = np.c_[np.ones(m),X]
    z2 =np.dot( a1 ,Theta1.conj().T);
    a2 = np.c_[np.ones((np.size(z2,0),1),dtype=float), sg.sigmoid(z2)];
    z3 = np.dot(a2,Theta2.conj().T);
    a3 = sg.sigmoid(z3)
    index_max = np.argmax(a3,axis=1)

    

    
    return index_max +1


"""

%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
    %
    %     lrCostFunction.m (logistic regression cost function)
    %     oneVsAll.m
    %     predictOneVsAll.m
    %     predict.m
    %
    %  For this exercise, you will not need to change any code in this file,
    %  or any other files other than those mentioned above.
    %

    

"""
#initialization
import numpy as np
import scipy.io
#Setup the parameters you will use for this exercise
input_layer_size  = 400;  #20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)
"""
=========== Part 1: Loading and Visualizing Data =============
 We start the exercise by first loading and visualizing the dataset. 
You will be working with a dataset that contains handwritten digits.
"""
#Load Training Data
print("\nLoading the data\n")

data=scipy.io.loadmat('ex3data1.mat')
X=data['X']
y=data['y'].astype(int)


m = np.size(X, 0);

#Randomly select 100 data points to display
sel =np.random.permutation(m);
sel = sel[0:100];
print('Program paused. Press enter to continue.\n')
"""
%% ================ Part 2: Loading Pameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.
"""

print('\nLoading Saved Neural Network Parameters ...\n')

#Load the weights into variables Theta1 and Theta2
weights=scipy.io.loadmat('ex3weights.mat');
Theta1=weights['Theta1']
Theta2=weights['Theta2']

import predict as pd
pred = pd.predict(Theta1, Theta2, X).reshape(m,1);

print('\nTraining Set Accuracy: %f\n'% np.mean(np.double(np.equal(pred , y).astype(int)) * 100));

print('Program paused. Press enter to continue.\n');

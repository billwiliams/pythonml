"""
%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
% 
%  This file contains code that helps you step through the following : 
%  i
    %
    %     lrCostFunction.m (logistic regression cost function)
    %     oneVsAll.m
    %     predictOneVsAll.m
    %     predict.m
    

"""
#initialization 
import numpy as np

import scipy.io

import oneVsAll as oneVsAll

input_layer_size  = 400;  # 20x20 Input Images of Digits
num_labels = 10; # % 10 labels, from 1 to 10   \
                  #        % (note that we have mapped "0" to label 10)

"""
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
"""
print("visualizing training  data");
data=scipy.io.loadmat('ex3data1.mat');# % training data stored in arrays X, y
#n=np.shape(X)
X=data['X']


y=data['y']

m=np.size(X,0)


rand_indices=np.random.permutation(m)



sel=X[rand_indices[0:100],:]

print(np.size(sel))
lamda=0.001
all_theta=oneVsAll.oneVsAll(X,y,num_labels,lamda)
import predictOneVsAll as pD
pred = pD.predictOneVsAll(all_theta, X);

accuracy=(np.mean(np.double(np.equal( y,pred).astype(int)))*100);
print('training accuracy is %f\n' %accuracy)


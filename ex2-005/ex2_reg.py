"""
% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.





"""

#importing some useful modules
import numpy as np

import pylab as pl

import plotData as pD
import mapFeature as mF
import predict as pd
import plotDecisionBoundary as pDB
data = np.loadtxt(('ex2data2.txt'),delimiter=",");
X = data[:, :2]; 
y = data[:, 2];


pD.plotData(X, y)
#labels and legends
pl.legend("10")

pl.xlabel('Microchip Test 1')
pl.ylabel('Microchip Test 2')
"""
%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled



"""
# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mF.mapFeature(X[:,0], X[:,1]);

#Initialize fitting parameters
m,n=np.shape(X)
initial_theta =np.zeros((n, 1),dtype=float);
labda = 1;
import costFunctionReg as cFR
cost,theta=cFR.costFunctionReg(initial_theta,X,y,labda)
pDB.plotDecisionBoundary(theta, X, y)

p = pd.predict(theta, X);

print('Train Accuracy: %f\n', np.mean(np.double(p == y)) * 100);

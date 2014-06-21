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
#numpy for linear algebra manipulations
import numpy as np
#pylab for plotting 
import pylab as pl
#plotData for plotting the training data in a 2-D plot
import plotData as pD
#mapFeature for mapping the features on our training set
import mapFeature as mF
#predict for predicting on our training  set
import predict as pd
#plotDecisionBoundary for plotting the curve used to fit the data
import plotDecisionBoundary as pDB
#scientific python for minimization of the cost function
from scipy import optimize
#loading data from the txt file to the variable data
data = np.loadtxt(('ex2data2.txt'),delimiter=",");
#the first two columns are assigned to x .They represent the features used
X = data[:,:2]; 
#y is the results i.e. 1 or 0
y = data[:,2];

#plotting the data on a 2-D plot
pD.plotData(X, y)
#labels and legends
pl.legend('10')

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
#initial guess
initial_theta =np.zeros((n, 1),dtype=float);
#regularization parameter .this can be changed
labda = 1.;
#importing optimization module and costfunction with regularization lambda
from scipy import optimize;
import costFunctionReg as cFR
#getting the cost from the function
cost=cFR.costFunctionReg(initial_theta,X,y,labda)
#optimizing using BFGS important to set jac=False i.e. jacobian is set to false
res = optimize.minimize(cFR.costFunctionReg, initial_theta, args=(X,y,labda), \
                                               method='BFGS',jac=False, options={'maxiter':400})
#theta and cost obtained from the cost function
theta=res.x
cost=res.fun
#plotting the decision boundary
pDB.plotDecisionBoundary(theta, X, y)
#adding some plotting elements
pl.xlabel('Microchip Test 1')
pl.ylabel('Microchip Test 2')
pl.title("lambda is %s "%labda)

pl.legend("10")
#predicting on our training set
p = pd.predict(theta, X);

print("Train Accuracy: %.4f%%  \n" %(np.mean(np.double(p == y)) * 100));

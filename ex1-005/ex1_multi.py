"""
%% python adaptation Machine Learning Online Class taught in coursera by professa Andrew ngi
%This code is not affiliated to coursera and neither is the author a coursera staff
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you step through the scripts
%  linear regression exercise. 
%
%  
%  exericise:
    %
    %     warmUpExercise.py
    %     plotData.py
    %     gradientDescent.py
    %     computeCost.py
    %     gradientDescentMulti.py
    %     computeCostMulti.py
    %     featureNormalize.py
    %     normalEqn.py
    %
    %  For this part of the exercise, you will need to change some
    %  parts of the code below for various experiments (e.g., changing
    %  learning rates).
    %

"""


#initialization

import numpy as np
#importing feature normalize as fN
import featureNormalize as fN
#importing gradient descent function for multiple features
import gradientDescentMulti as gDM
#importing plotting functionality
import pylab as plt
#import normal equation function
import normalEqn as nEqn
"""   %% ================ Part 1: Feature Normalization ================ """
print('Loading data ...\n');

# Load Data
data = np.loadtxt(('ex1data2.txt'),delimiter=",");
X = data[:, 0:2]
y = data[:, 2:3];
m = len(y);

#Print out some data points
print('First 10 examples from the dataset: \n');
print('  X \n %s\n   y\n %s \n'  %(X[0:10,:] ,y[0:10,:]));

print('Program paused. Press enter to continue.\n')
raw_input()
# Scale features and set them to zero mean
print('Normalizing Features ...\n');

#getting back the mu,sigma and X_norm from the tuple gotten from the feature normalize
mu,sigma,X=fN.featureNormalize(X)
print("%s %s " %(mu,sigma))
# Add intercept term to X
X=np.c_[np.ones(m),X]

"""% ================ Part 2: Gradient Descent ================"""
print('Running gradient descent ...\n');

# Choose some alpha value
alpha = 0.1
num_iters = 400
#initializing theta to zeros
theta=np.zeros((3,1),dtype=float)
#converting from a numpy array to a matrix


#running gradient descent with multiple features

theta,J_history=gDM.gradientDescentMulti(X,y,theta,alpha,num_iters)

#Plot the convergence graph
plt.ion()
fig=plt.figure()
#setting up x dimension matlab uses nume1 to get number of array elements
xdimension=np.arange(len(J_history))
xdimension=xdimension.reshape(len(J_history),1)

plt.plot(xdimension, J_history, '-b',linewidth=2.0);
#setting up the scales
plt.xlim([-3,400])

plt.xlabel('Number of iterations');
plt.ylabel('Cost J');


#Display gradient descent's result
print('Theta computed from gradient descent: \n');
print(" %s \n" %theta);
print('\n');
price =np.matrix( [ 1 , -0.44604386, -0.224428357 ]) * theta
price=str(price).replace('[','').replace(']','')
print("Predicted price of 1650sq with 3 bedrooms is %s $\n"%price)

"""==============================Normal Equation ==================="""
#load Data since we dont normalize it in Norma Equation
data = np.loadtxt(('ex1data2.txt'),delimiter=",");
X = data[:, 0:2]
y = data[:, 2:3];
m = len(y);
# Add intercept term to X
X=np.c_[np.ones(m),X]

#Display theta computed from norma equations

print('Theta computed from the normal equations: \n');
theta=nEqn.normalEqn(X,y)
print(' %s \n' %(theta));
print('\n');
price =np.matrix( [1,  1650, 3] )* theta

price=str(price).replace('[','').replace(']','')
"""==========================predicting price of a house using normal Eqns=================================="""

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n%s $\n' %price);

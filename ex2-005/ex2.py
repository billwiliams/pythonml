"""
%% Python adaptation Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you step through the functions for logistic
%  regression exercise. 




"""

#initialization

#importing neccesary modules
#module for linear algebra computations
import numpy as np
#for plotting
import pylab as pl
#import plotData function as pD
import plotData as pD
#importing costFunction as cF
import costFunction as cF
#importing MapFeature
import mapFeature as mF
#importing sigmoid fucntion
import sigmoid as sg
#loading the training data from text file
data=np.loadtxt(("ex2data1.txt"),delimiter=",")
X= data[:,:2]
y = data[:, 2]
"""=========================plotting=========================="""

# We start the exercise by first plotting the data to understand the   the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
#plotting the Data
pD.plotData(X, y);

#Put some labels ;


pl.show()
pl.xlabel('Exam 1 score')
pl.ylabel('Exam 2 score')



print('\nProgram paused. Press enter to continue.\n');

raw_input()


#getting the size of matrix X in the form [m,n]
m=X.shape[0]
n=X.shape[1]
#Adding an intercept column of ones to the matrix 

X= np.column_stack((np.ones(m), X))
#initial theta
initial_theta= np.zeros(n + 1)
cost,grad=cF.costFunction(initial_theta,X,y)
print('Cost at initial theta (zeros): %s\n'% cost);
print('Gradient at initial theta (zeros): \n');
print(' %s \n'% grad);

print('\nProgram paused. Press enter to continue.\n');
raw_input()
"""

%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc

"""
#importing fmin_ncg from  s
from scipy import optimize

# optimizing   costfunction using Newton_CG

res = optimize.minimize(cF.costFunction, initial_theta, args=(X,y), \
                                                method='Newton-CG', jac=True, options={'maxiter':400})
#theta obtained by the Newton-CG optimization minimizing technique
theta=res.x
#Cost obtained by the Newton-CG
cost=res.fun
#printing the values
print("theta found by Newton-CG is %s\n" %theta)
print("cost found by Newton-CG is %s\n" %cost)

"""
============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%



"""
#importing plotting decision boundary
import plotDecisionBoundary as pDB
pDB.plotDecisionBoundary(theta,X,y)
#reshaping theta to a matrix of 3X1 inorder to compute matrix multiplication
thet=np.matrix(theta.reshape(3,1))

prob =sg.sigmoid([1, 45, 85] * thet);
prob=str(prob).replace(' ','').replace('[','').replace(']','')
print('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
#predicting on our data
import predict as pd
p = pd.predict(theta, X)
#testing the accuracy of the prediction
print( 'Train Accuracy: %.1f%%' % ((p == y).mean() * 100))

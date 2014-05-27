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

#loading the training data from text file
data=np.loadtxt(("ex2data1.txt"),delimiter=",")
X=data[:,0:2]
y=data[:,2:3]
"""=========================plotting=========================="""

print('Plotting data with + indicating (y = 1) examples and o ......\n indicating (y = 0) examples.\n')

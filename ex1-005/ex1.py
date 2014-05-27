""" ===========Machine Learning Online Class - Exercise 1: Linear Regression==================================
%This is the python adaptation of the coursera course taught by prof Andrew Ng
%This work is not property of coursera and neither is the Author associated with coursera staff

%  Instructions
%  ------------
% 
%  This file contains code for linear regression with single feature and helps step through the steps
%  linear exercise.  
%  in this exericse:
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
    %  
    %
    % X refers to the population size in 10,000s
    % y refers to the profit in $10,000s
    %

   ============================================== ipython shell is recommended since plotting using pylab is far much better supported===========
  
    ....
    """
#initialization
#import numpy for matrix manipulations
import numpy as np
#importing pylab for plotting
import pylab as pl
#import 3d axes plots from matpotlib
from mpl_toolkits.mplot3d import Axes3D
#importing plotData file with the function for ploting the curves and we access it use initials plt
import plotData as plt

#importing gradientDescent as gD
import gradientDescent  as gD
#importing warmUpExercise file with  identity matrix function
import warmUpExercise as wmUpEx
#import computeCost file with computeCost function and to access it we use initials as cC
import computeCost as cC
print("Running warmUpExercise ... \n")
print("5x5 Identity Matrix: \n")
#warmUpExercise to print the identity matrix similar to eye(5) in octaveor matlab
A=wmUpEx.warmUpExercise()

print(A)#printing the identity matrix
print("Program paused press enter to continue ")

"""%% ======================= Part 2: Plotting ======================= """
#load data located in ex1data1 in a variable called data
data=np.loadtxt('ex1data1.txt',delimiter=",")
#load data in the first column of variable data in variable X 
#unlike in matlab in python indexing starts at 0
X = data[:, 0]

#load data in second column of variable data in variable y
y = data[:, 1]
#get the length of y
m=len(y)
#set y as a m*1 matrix .this is because numpy stores it as a numpy array and to perform linear algebra we need as an m by1 matrix 

y=y.reshape(m,1)


#calling the function to plot data 

ax=plt.plotData(X,y)
"""%% =================== Part 3: Gradient descent =================== """
#initializing theta to zeros that is for initial values for theta we set it to zeros with a data type float
theta=np.zeros((2,1),dtype=float)

#adding a ones column to X so that we can use the preceeding column as a feature
X=np.c_[np.ones(m),X]


#compute the cost of the initial values
J=cC.computeCost(X, y,theta)
#setting variables needed by the gradient descent which requires X,theta,y alpha ,num_iters
iterations = 1500;
alpha = 0.01;
#printing the cost with one variable when theta is initialized to zeros .it should be approximately 32.07
print("the cost should be approximately equal to 32.07 \n %s"%(J))

#calculating the gradientDescent
[theta,J_history ] = gD.gradientDescent(X, y, theta, alpha, iterations);
#dot multiplication of the array same as X* theta in matlab
df=np.dot(X,theta)
#plotting the data for the linear regression curve
pl.plot((X[:,1]), (df[:,0]), '-') 
pl.legend('dt')
pl.ion()
pl.show(ax)#former data plot
raw_input()

#setting the theta array
theta=np.array(theta)
#reshaping array to form a matrix
theta=theta.reshape(2,1)
#setting the features of the house to 1,3.5
predict =np.matrix([1, 3.5]) 
predict1=predict* theta
#price1=price of the house with the above features
price1=predict1* 10000
#removing the numpy brackets
price1=str(price1).replace(' ','').replace('[','').replace(']','')
print("For population = 35,000, we predict a profit of %s $ \n " %(price1))
#setting the matrix for the features of house to predict
predict =np.matrix([1, 7])
#predicting the value of a house with the features 1 7
predict2=((predict* theta)*10000)
#removing the numpy brackets in the array
price2=str(predict2).replace(' ','').replace('[','').replace(']','')
print("For population = 70,000, we predict a profit of %s $ \nPress Enter to continue\n" %(price2))

raw_input()
"""%% ============= Part 4: Visualizing J(theta_0, theta_1) ============= """
print('Visualizing J(theta_0, theta_1) ...\n')

#Grid over which we will calculate J
theta0_vals =( np.linspace(-10, 10, 100));
theta1_vals = (np.linspace(-1, 4, 100));
#getting the length of the each grid
theta0_len=len(theta0_vals)
theta1_len=len(theta1_vals)
#initialize J_vals to a matrix of 0's
J_vals = np.zeros([theta0_len, theta1_len]);


for i in range(theta0_len):
    for j in range(theta1_len):
        t = [theta0_vals[i], theta1_vals[j]]
        t=np.array(t)
        t=t.reshape(2,1)
    
        J_vals[i,j] = cC.computeCost(X, y, t)




#plotting the surface plot 
fig = pl.figure()
#since the plot is in 3d we assign projection to 3d as illustrated in axes3d imported
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals,np.transpose(J_vals), rstride=8, cstride=8, alpha=0.3)
print("press Enter to continue ......\n")
raw_input()
#create another figure for contour plotting
fig=pl.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
pl.contour(theta0_vals, theta1_vals, np.transpose(J_vals), np.logspace(-2, 3, 20))

#plot the value of theta to determine if the cost was minimized
pl.plot(theta[0],theta[1],'rx')

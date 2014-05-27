PYTHONML
==================


Intro
-----

PYTHONML is the python adaptation of the Machine Learning course on coursera taught by proffesa Andrew Ng.

The following modules and tools are recquired to have the same experience as matlab codes provided by coursera:
	1.python numpy module for linear algebra computations

	*Ipython shell for running the scripts
	
	*Pylab for plotting the curves and graphs 
 
### Exercise 1 :Linear Regression ###
 This exercise contains files for linear regression with one variable/feature and also with mutiple variables/feature.
  **linear regression with one variable**
   1. ex1.py helps step through the linear regression with one variable/feature
   + ex1data1.txt file contains data to be used for linear regression with one variable
   + warmUpExercise.py contains code that generates a 5x5 identity matrix
   + plotData.py contains code for plotting the data provided for the house prices and population
   + computeCost.py contains code that calculates the cost of using theta
   + gradientDescent.py contains the code for computing theta by simultaneous update and returns the values of theta
	
  ***linear regression with multiple variables***
   1. ex1_multi.py helps step through the scripts for computing linear regression with multiple variables
   * ex1data2.txt file contains data for the linear regression with multiple variables
   * computeCostMulti.py file contains code for calculating the cost of using theta
   + featureNormalize.py contains the code for normalizing each feature for the data given
   + gradientDescent.py contains the code for updating theta on each iteration i.e.
   + 
--- 
   for iter in range(0,num_iters):
        Xtheta=np.dot(X,theta)
        thetaminusy=Xtheta-y
        theta-=np.dot((alpha/m),np.dot(Xtranspose,thetaminusy))

---
   + normalEqn.py contains code to calculate theta using normal equations :


#### Linear regression algorithm usage ####
   1. **Finance**
	-The capital asset pricing model uses linear regression for analyzing and quantifyingthe systematic risk of an investment
   2. **Economics**
	-Linear regression is used to predict consumptional spending,fixed investment spending and inventory investments e.t.c
   3. **Enviromental science**
	-The Enviromental Effects Monitoring Program in canada uses statiscal analysis on fish and bethnic surveys to measure the effets of pulp mill 		or metal mine affluent on the acquatic ecosystem.


** Credits **
  *This code was edited and compiled by [Ndirangu Wilson](wilson.ndirangu@myshoppingmate.com) *




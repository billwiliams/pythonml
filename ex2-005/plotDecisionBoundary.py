"""
function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones







"""

def plotDecisionBoundary(theta,X,y):
    #importing some useful modules
    import plotData as pD
    #linear algebra computations using numpy
    import numpy as np
    #plotting module
    import pylab as plt
    import mapFeature as mF
    # plot the Data
    pD.plotData(X[:,1:3],y)
    #getting the shape of the matrix x
    m,n=np.shape(X)
    plt.ion()

    if (n<=3):
        #Only need 2 points to define a line, so choose two endpoints
        plot_x= np.min(X[:,1])-2,np.max(X[:,1])+2
        #plot 
        plot_y=np.dot((-1./theta[2]),np.dot(theta[1],plot_x)+theta[0])
        plt.plot(plot_x, plot_y)
    else:
         u =np.linspace(-1, 1.5, 50);
         v = np.linspace(-1, 1.5, 50);

         z = np.zeros((len(u), len(v)),dtype=float);

         for i in range (len(u)):
             for j in range(len(v)):
                 z[i,j]=mF.mapFeature(u[i], v[j])*theta
         z=z.T

         plt.contour(u, v, z, [0, 0], 'LineWidth=2.0')


        
    



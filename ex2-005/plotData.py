"""

%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.


% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
"""
#function to plot data
def plotData(X,y):
    #importing numpy and pylab for plotting the data
    import numpy as np
    import pylab as pl
    pos=X[np.argwhere(y)]
    #pos=pos.reshape([:,1])
    neg=X[np.where(y==0)[0][0]]
    #neg=neg.reshape([:,1])
    pl.ion();
    fig=pl.figure();
    pl.plot(X[pos, 0], X[pos, 1], 'k+')
    return [pos,neg]

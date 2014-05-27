"""
%function plotData(x, y)
%
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.
%
% 
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands using pylab. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses.


"""
def plotData(x,y):
    import pylab  as pl
    pl.ion() #start interactive mode in pylab to enable script continuation after showing the figure
    pl.plot(x, y, 'rx'); # Plot the data using red x marks
    pl.xlim([4,24])
    pl.ylim([-5,25])
    pl.ylabel('Profit in $10,000s'); # Set the y?axis label
    pl.xlabel('Population of City in 10,000s'); # Set the x?axis label
    
    pl.show() #show the plot
    
    raw_input() #wait for pause
    pl.ioff()
    return

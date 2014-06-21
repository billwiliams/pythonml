"""
 Machine Learning Online Class - Exercise 4 Neural Network Learning

 %  Instructions
 %  ------------
 % 
 %  This file contains code that helps you get started on the
 %  linear exercise. You will need to complete the following functions 
 %  in this exericse:
     %
     %     sigmoidGradient.m
     %     randInitializeWeights.m
     %     nnCostFunction.m
"""
#importing some useful modules
import numpy as np
import scipy.io
import nnCostFunction as nn
import sigmoidGradient as sG
import randInitializeWeights as rIW
#Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10  (note that we have mapped "0" to label 10)
print("visualizing training  data");
data=scipy.io.loadmat('ex4data1.mat');# % training data stored in arrays X, y

X=data['X']


y=data['y']

m=np.size(X,0)


rand_indices=np.random.permutation(m)



sel=X[rand_indices[0:100],:]

print(np.size(sel))

#load the weigths into Theta1 and Theta2
weights=scipy.io.loadmat('ex4weights.mat');
Theta1=weights['Theta1']
Theta2=weights['Theta2']

#Unroll parameters 
nn_params =np.r_[Theta1.flatten(1),Theta2.flatten(1)]
"""
%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
"""
print("\nFeedforward Using Neural Network ...\n")

# Weight regularization parameter (we set this to 0 here).
lamda = 0.0;
J,grad = nn.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda);
print(" The cost of at parameters (loaded from ex4weights): %f\n" %J)
print('\nProgram paused. Press enter to continue.\n');
raw_input();
"""
%% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%
"""
print('\nChecking Cost Function (w/ Regularization) ... \n')

#Weight regularization parameter (we set this to 1 here).
lamda = 1.0;

J,grad = nn.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda);

print('Cost at parameters (loaded from ex4weights): %f \n' %J);

print('Program paused. Press enter to continue.\n');
raw_input();
"""
%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%
"""
print('\nEvaluating sigmoid gradient...\n')

g = sG.sigmoidGradient(np.array([1 ,-0.5, 0 ,0.5, 1]));
print('Sigmoid gradient evaluated at [1., -0.5, 0. ,0.5, 1.]:\n  ');
print('%s ' %g);
print('\n\n');

print('Program paused. Press enter to continue.\n');
raw_input();
"""
%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.py)
"""

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = rIW.randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = rIW.randInitializeWeights(hidden_layer_size, num_labels);

#Unroll parameters
initial_nn_params =np.r_[initial_Theta1.flatten(1) , initial_Theta2.flatten(1)];

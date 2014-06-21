def sigmoid(z):
    """
    a Function to return the sigmoid given a certain value
    the calculation is as 1.0/1.0+exp(-z)
    """
    import numpy as np
    g=1.0/(1.0+np.exp(-z))
    return g

def predict(theta, X):
    
        """
        Return predictions for a set of test scores.
        """
        import sigmoid as sg
        import numpy as np

        p = sg.sigmoid(np.dot(X, theta)) >= 0.5
        return p

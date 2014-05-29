





import numpy as np
def mapFeature(X, degree=1):
    """
    Take X, an m x 2 DataFrame, and return a numpy array with more features,
    including all degrees up to degree.
    1, X1, X2, X1^2, X2^2, X1*X2, X1*X2^2, etc.
    """

    m, n = np.shape(X)

    if not n == 2:
	    raise ValueError('mapFeature supports input feature vectors of length 2, not %i' % n)

    out = np.ones([1, m])

    for totalPower in xrange(1, degree+1):
         for x1Power in xrange(0, totalPower+1):
                out = np.append(out, [X[0]**(totalPower-x1Power) * X[1]**x1Power], axis=0)

    return np.transpose(out)

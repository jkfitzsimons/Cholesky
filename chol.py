__author__ = 'jackfitzsimons'

# Cholesky Decomposition
#
# A simple iterative version of the cholesky decomposition
# adding the minimum level of jitter to avoid singular matrix
# calculations - assuming python float calculations
#

import numpy as np
import sys

def chol(A):

    # Determine the Cholesky decomposition for the positive semi-definite matrix A
    # Add the minimum amount of jitter allowed by python float

    # Initialize
    X = A.copy()
    n = X.shape[1]
    
    # Epsilon as defined by float type
    eps = sys.float_info.epsilon

    # Main algorithm
    for k in range(0, n):
        for i in range(0, k):
            s = 0.0
            for j in range(0, i):
                s = s + (X[i,j]*X[k,j])
            X[k, i] = (X[k, i] - s)/X[i, i]
        s = 0.0
        for j in range(0, k):
            s = s + pow(X[k,j], 2)
        if X[k,k] == s:
            # add jitter when necessary
            X[k,k] = X[k,k]+eps
        X[k,k] = np.sqrt(X[k,k] - s)

    return np.tril(X, 0)

X = np.array(((1,1,1),(1,1,1),(1,1,2)), dtype=float)

try:
    print(np.linalg.cholesky(X))
except:
    print('np.linalg.cholesky(X) failed')

print(chol(X))
print(np.dot(chol(X),chol(X).T))

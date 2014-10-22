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
                s += (X[i,j]*X[k,j])
            X[k, i] = (X[k, i] - s)/X[i, i]

        s = 0.0
        for j in range(0, k):
            s += pow(X[k, j], 2)
        if s >= X[k,k]:
            # add jitter when necessary
            print('noise added: ' + str(s-X[k,k]) + ' to input ' + str(k))
            X[k,k] = eps
        else:
            X[k, k] -= s
            X[k,k] = np.sqrt(X[k,k])




    return np.tril(X, 0)

X = np.array(((2,1,1,1),(1,2,1,1),(1,1,2,1),(1,1,1,2)), dtype=float)

r = np.linalg.matrix_rank(X)

print(r)
print(np.min(np.linalg.eigvals(X)))

try:
    print(np.linalg.cholesky(X))
except:
    print('np.linalg.cholesky(X) failed')

print(chol(X))
out = np.dot(chol(X),chol(X).T)
print(out)
r = np.linalg.matrix_rank(out)
print(r)
print(np.min(np.linalg.eigvals(out)))

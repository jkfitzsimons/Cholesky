__info__ = '''
Author: Jack Fitzsimons
Affiliation: Machine Learning Research Group, University of Oxford
Supervisors: M. Osborne, S. Roberts
Code: An implementation of a safe cholesky decomposition
        which adds the minimum amount of jitter to the
        minimum number of inputs in order to make it
        positive definite.
Platform: Written for MacOSX with Accelerate Framework (cBlas)
'''



c_source = '''
    X[0] = sqrt(X[0]);

    double s = 0;
    long d_size = &X[1] - &X[0];
    double* X_local_1;
    double* X_local_2;
    int len;

    for(int k = 1; k < n; k++) {
        // Evaluate the off diagonals
        for(int i = 0; i < k; i++) {
            s = 0;
            X_local_1 = X+(i*n*d_size);
            X_local_2 = X+(k*n*d_size);
            len = i;
            s = cblas_ddot(len, X_local_1, 1, X_local_2, 1);

            X[(k*n)+i] = (X[(k*n)+i] - s)/X[(i*n)+i];
        }

        // Evaluate the diagonals
        int len = k;
        X_local_1 = X+(k*n*d_size);
        s = cblas_ddot(len, X_local_1, 1, X_local_1, 1);
        if(s >= X[(k*n)+k]){
            X[(k*n)+k] = 0.000000000001;
        }
        else{
            X[(k*n)+k] = sqrt(X[(k*n)+k]-s);
        }

    }

    for(int k = 0; k < n-1; k++) {
        for(int i = k+1; i < n; i++) {
            X[(k*n)+i] = 0;
        }
    }
    return_val = X;

'''

import scipy.weave as wv
import scipy as sp
import numpy as np
import time
import pylab as pb
import matplotlib.pyplot  as plt
from distutils.extension import Extension


'''
The rest of the code merely wraps it in scipy weave to make it easily called
and was used for debugging / testing purposes.
'''

X = np.ones((1000,1000))
n = X.shape[0]
Z = X.copy()
wv.inline(c_source,['X', 'n'], headers = ["<math.h>", "<Accelerate/Accelerate.h>"], extra_link_args = ["-framework Accelerate"], force=1)
print(np.sum(abs(Z-sp.dot(X, X.T))))


t_old = []
t_new = []
for i in range(1,41):
    X = np.random.rand(i*100,1)

    X = np.exp(-pow((X - X.T), 2)) + (np.eye(i*100)*10000)

    n = X.shape[0]

    start_time = time.time()
    Y = np.linalg.cholesky(X)
    t_old.append(time.time()-start_time)

    start_time = time.time()
    wv.inline(c_source,['X', 'n'], headers = ["<math.h>", "<armadillo>", "<iostream>", "<Accelerate/Accelerate.h>", "<typeinfo>"], extra_link_args = ["-framework Accelerate"])
    #X = sp.tril(X.T)
    t_new.append(time.time()-start_time)



plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

line_1, = ax.plot(range(100,4001, 100), pow(np.array(t_old), 0.333), color='blue')
line_2, = ax.plot(range(100,4001, 100), pow(np.array(t_new), 0.333), color='red')

#ax.set_yscale('log')

np.savetxt('t_old.txt', t_old, delimiter=',')
np.savetxt('t_new.txt', t_new, delimiter=',')

pb.show(block=True)






c_source = '''
    using namespace arma;
    mat M = mat(X, n, n, false, true);

    M(0,0) = sqrt(M(0,0));

    double s;

   for(int k = 1; k < n; k++) {
        // Evaluate the off diagonals
        for(int i = 0; i < k; i++) {
            if(i>0){
                s = sum(sum(M.submat(i,0,i,i-1)%M.submat(k,0,k,i-1)));
            }else{
                s = 0;
            }
            M(k,i) = (M(k,i) - s)/M(i,i);
        }

        // Evaluate the diagonals
        s = sum(sum(square(M.submat(k,0,k,k-1))));
        if(s >= M(k,k)){
            M(k,k) = 0.000000000001;
        }
        else{
            M(k,k) = sqrt(M(k,k)-s);
        }
    }

    return_val = X;

'''



__author__ = 'jackfitzsimons'

import scipy.weave as wv
import scipy as sp
import numpy as np
import time
import pylab as pb
import matplotlib.pyplot  as plt
from distutils.extension import Extension

t_old = []
t_new = []
for i in range(1,151):
    X = np.random.rand(10*i,1)

    X = np.exp(-pow((X - X.T), 2)) + (np.eye(10*i)*10000)

    n = X.shape[0]

    start_time = time.time()
    Y = np.linalg.cholesky(X)
    t_old.append(time.time()-start_time)
    #print('Standard cholesky time: ' + str(time.time()-start_time))

    start_time = time.time()
    wv.inline(c_source,['X', 'n'], headers = ["<math.h>", "<armadillo>", "<iostream>"])
    X = sp.tril(X.T)
    t_new.append(time.time()-start_time)
    #print('New cholesky time: ' + str(time.time()-start_time))

    print(i*10)



#print(np.mean(abs(sp.dot(Y, Y.T) - sp.dot(X, X.T))))
#print(np.max(abs(sp.dot(Y, Y.T) - sp.dot(X, X.T))))


#print(Y)
#print(' ')
#print(X)
#print(' ')
#print(sp.dot(Y, Y.T))
#print(' ')
#print(sp.dot(X,X.T))

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(2,1,1)

line_1, = ax.plot(range(10,1501, 10), pow(np.array(t_old), 0.333), color='blue')
line_2, = ax.plot(range(10,1501, 10), pow(np.array(t_new), 0.333), color='red')

#ax.set_yscale('log')

np.savetxt('t_old.txt', t_old, delimiter=',')
np.savetxt('t_new.txt', t_new, delimiter=',')

pb.show(block=True)






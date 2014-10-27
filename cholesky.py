__info__ = '''
Author: J. Fitzsimons
Affiliation: Machine Learning Research Group, University of Oxford
Supervisors: M. Osborne, S. Roberts

Code: An implementation of a safe cholesky decomposition which adds the
        minimum amount of jitter to the minimum number of inputs in order
        to make it positive definite.

     When performing the cholesky decomposition we find ourselves
        performing a square root of a negative value iff the matrix
        is not positive definite. This occurs while updating the diagonal.
        Due to the nature of the cholesky decomposition algorithm, we can
        find this before we ever use the diagonal value to solve for
        other elements of the resulting lower triangular matrix.

     Using this insight we can identify if there is a row/column which causes
        the matrix to be singular and if so add the minimum amount of 'jitter'
        to its diagonal element to force the matrix to be positive semi definite.

     The codes performance was built and tested on a MacBook Pro running OSX 10.9.5
        and has been seen to run at roughly 60% of the speed of numpys
        np.linalg.cholesky(). The discrepancy in speed is due to the additional
        cost in the cBLAS c-api versus LAPACKs direct access.

Platform: Written for MacOSX with Accelerate Framework (cBlas)
'''

import scipy.weave as wv
import numpy as np

def safe_chol(Y):
    # The C source for the cholesky decomposition with cBLAS
    c_source = '''
        X[0] = sqrt(X[0]);

        // Used to
        double s;


        long d_size = &X[1] - &X[0];

        // Pointers will be used to reference start of columns
        // for matrix arithmatic
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
                // Force matrix to be PSD
                X[(k*n)+k] = DBL_MIN;
            }
            else{
                // This sqrt() will fail if not PSD
                X[(k*n)+k] = sqrt(X[(k*n)+k]-s);
            }

        }

        // Zero out upper triangle of matrix
        for(int k = 0; k < n-1; k++) {
            for(int i = k+1; i < n; i++) {
                X[(k*n)+i] = 0;
            }
        }

    '''

    # The rest of the code merely wraps the c code in scipy weave and calls the function

    # Make a copy of X so that we don't make changes to it
    X = np.copy(Y)

    # Find the dimensionality of Y
    n = X.shape[0]

    # Confirm that we are dealing with a square matrix
    if n != X.shape[1]:
        print('Matrix is not square, invalid arguments.')
        return None

    # If you are using a Windows or Linux machine please edit appropriate header file and framework arguments to link cBLAS
    wv.inline(c_source,['X', 'n'], headers = ["<math.h>", "<Accelerate/Accelerate.h>"], extra_link_args = ["-framework Accelerate"])

    return X


# This code simply allows you to test the functionality of safe_chol()
x = np.ones((3,3)) + np.eye(3)
print('Simple cholesky decomposition of np.ones((3,3)) + np.eye(3):')
print(safe_chol(x))
print('Reconstruction of np.ones((3,3)) + np.eye(3):')
print(np.dot(safe_chol(x),safe_chol(x).T))


x = np.ones((3,3))
print('Simple cholesky decomposition of np.ones((3,3)):')
print(safe_chol(x))
print('Reconstruction of np.ones((3,3)):')
print(np.dot(safe_chol(x),safe_chol(x).T))

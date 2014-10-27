Cholesky
========

Cholesky Decomposition 

A sequential cholesky decomposition with run time O(n^3). The routine
identifies when the matrix appears to be singular and adds epsilon
jitter to the diagonal of these rows. Epsilon is defined by the double
type in c++.

The reason our covariance matrices often appear singular is because 
two or more rows (or linear combinations of these) are identical to machine 
precision. To avoid this it is common to add iid noise to the entire 
covariance matrix. However this is an unnecessary hack to a numerical 
precision issue.

The code provided identifies this numerical issue occurs and adds the 
minimum amount noise to the diagnol to create a numerically stable result.

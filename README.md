# iADMM
An Inertial Alternating Direction Method of Multipliers (iADMM) for a low-rank representation optimization problem
min sum_ig(\sigma_i(Z))+ p(E) + r(Y)
s.t., X = AZ+ EB + Y 
inputs:
X -- D*N data matrix, D is the data dimension, and N is the number of data points.

Reference: LTK Hien, DN Phan, N Gillis. "A Framework of Inertial Alternating Direction Method of Multipliers for Non-Convex Non-Smooth Optimization".

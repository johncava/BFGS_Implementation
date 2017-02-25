from __future__ import division
import numpy as np

def function(x):
    return x**2

def function_deriv(x):
    return 2*x

function_vec = np.vectorize(function)
function_deriv = np.vectorize(function_deriv)

#Hessian Matrix set to Identity Matrix
B = np.array( [ [1,0],
                [0,1] ])

#Pick initial point x_0
x_k = np.array([2,1])

#Obtain a direction p_k by solving B_k*p_k = -function_deriv(x_k)
p_k =  -np.linalg.inv(B).dot(function_deriv(x_k))

#Perform line search to find acceptable stepsize alpha_k such that x_k+1 = x_k + alpha_k*p_k
#-------- "min s |--> f( x_0 + s*p_0)"
#--------  --> Newton(once)
#------------- s_n+1 = s_n - g(s_n)/g'(s_n)
#------------- s_0 = 0 (initial s_0 condition, could be anything)
g = function_deriv(x_k)
#Calculate step size alpha
alpha_k = -(x_k[0] ** 2 + x_k[1] ** 2)/(g[0] + g[1])
#Set s_k = alpha_k*p_k
s_k = alpha_k*p_k
#Set x_k+1 = x_k + alpha_k*p_k
x_k1 = x_k + alpha_k*p_k
#Set y_k = f'(x_k+1) - f'(x_k)
y_k = function_deriv(x_k1) - function_deriv(x_k)
#Calculate B_k+1
B_k1 = B + (y_k.dot(y_k.T))/(y_k.T.dot(s_k)) - (B.dot(s_k.dot(s_k.T.dot(B))))/(s_k.T.dot(B.dot(s_k)))
print B_k1

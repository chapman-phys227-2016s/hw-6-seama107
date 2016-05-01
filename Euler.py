#!\usr\bin\python
"""
File: Euler.py

Copyright (c) 2016 Michael Seaman

License: MIT

As part of HW6, this implementation aids with solving the ODE's
given as the problems
"""

import numpy as np

def Euler(dydx, y_a, delta_x, a, b):
    """
    Takes the function dydx and computes an approximation of
    the function it models using Euler's method and a stepsize
    delta_x from a to b
    """
    x = np.arange(a, b, delta_x, dtype = np.float64)
    y = np.zeros(len(x)) + y_a
    for i in xrange(1,len(y)):
        y[i] = y[i-1] + delta_x * dydx(x[i-1], y[i-1])
    return y

def linear_derivative(x, y):
    return -.25

def linear_function(x, y):
    return -.25 * x + 20

def test_Euler():
    """
    Implemented around the fact that Euler's method will approximate
    a linear curve perfectly.
    """
    a = -20
    b = 30
    d_x = .001
    y_a = linear_function(a,6)
    xList = np.arange(a, b, d_x)
    lf_vec = np.vectorize(linear_function)
    exact = lf_vec(xList, xList)
    approx = Euler(linear_derivative, y_a, d_x, a, b)
    assert np.allclose(exact, approx)
    

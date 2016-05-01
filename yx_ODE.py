#!\usr\bin\python
"""
File: yx_ODE.py

Copyright (c) 2016 Michael Seaman

License: MIT

Excercise C.3:
Implements the derivative of a function y(x) to test
how well the Euler method approximates it
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from Euler import Euler


def dydx(x, y):
    return (2*y - 2)**-1

def plot_Euler_approx(dydx, y_a, delta_x, a, b):
    """
    Plots the Euler approximation of y(x) given function dfdx and
    the initial point y_a from a to b with stepsize delta_x
    """
    xList = np.arange(a, b, delta_x)
    yList = Euler(dydx, y_a, delta_x, a, b)
    plt.plot(xList, yList)
    plt.show()

def y_0(epsilon):
    """
    returns the initial condition given an epsilon value
    """
    return 1 + float(epsilon)**.5

def plot_approx_and_exact(exact, resolution, dfdx, y_a, delta_x, a, b):
    """
    Plots the Euler approximation y(x) in red alongside its 
    given exact value in green from a to b
    """
    xList_a = np.arange(a, b, delta_x)
    yList_a = Euler(dfdx, y_a, delta_x, a, b)
    xList_e = np.linspace(a, b, resolution)
    yList_e = np.vectorize(exact)(xList_e)
    plt.plot(xList_a, yList_a, 'r')
    plt.plot(xList_e, yList_e, 'g')
    plt.show()




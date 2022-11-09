import numpy as np

def step_function(x):
    return np.sign(x)

def branin(x):

    assert len(x.shape) == 2 

    if x.shape[0] != 2: x = x.T
    if x.shape[0] != 2: raise ValueError("The shape of x is not 2D.")

    x1 = x[0]
    x2 = x[1]

    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)

    return a*((x2 - b*(x1**2) + c*x1 - r)**2) + s*(1 - t)*np.cos(x1) + s

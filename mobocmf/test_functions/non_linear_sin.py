import numpy as np

def non_linear_sin_mf1(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * non_linear_sin_mf0(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd

def non_linear_sin_mf0(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd
    
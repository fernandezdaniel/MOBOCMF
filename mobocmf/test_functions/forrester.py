import numpy as np

def forrester_mf1(x, sd=0):
    """
    Forrester function

    :param x: input vector to be evaluated
    :param sd: standard deviation of noise parameter
    :return: outputs of the function
    """
    x = x.reshape((len(x), 1))
    n = x.shape[0]
    fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    if sd == 0:
        noise = np.zeros(n).reshape(n, 1)
    else:
        noise = np.random.normal(0, sd, n).reshape(n, 1)
    return fval.reshape(n, 1) + noise

def forrester_mf0(x, sd=0):
    """
    Low fidelity forrester function approximation:

    :param x: input vector to be evaluated
    :param sd: standard deviation of observation noise at low fidelity
    :return: outputs of the function
    """
    high_fidelity = forrester_mf1(x, 0)
    return (0.5 * high_fidelity + 10 * (x[:, [0]] - 0.5) + 5 + np.random.randn(x.shape[0], 1) * sd)
    

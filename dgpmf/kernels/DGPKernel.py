import math
import torch
import gpytorch
from matplotlib import pyplot as plt

class CustomKernel(gpytorch.kernels.Kernel): # Necesita saber que inputs seon los de la capa anterior y cuales los inputs externos
    # the sinc kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, **params): # Concatenar los x y los f de las diferentes capas
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)


import math
import torch
from gpytorch.kernels import Kernel, RBFKernel, LinearKernel
from gpytorch.kernels import RBFKernel
from matplotlib import pyplot as plt
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode

from ..kernels.WhiteNoiseKernel import WhiteNoiseKernel

class MFDGPKernel(Kernel): # Necesita saber que inputs son los de la capa anterior y cuales los inputs externos
    # the sinc kernel is stationary
    is_stationary = True

    def __init__(self, length_prior=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)

        Din = X[0].shape[1]
        Dout = Y[0].shape[1]

        kernels = []
        k_2 = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims) # (Din, active_dims=list(range(Din)), variance=1., lengthscales=10., ARD=True)
        kernels.append(k_2)
        for l in range(1,L):
            
            D = Din + Dout
            D_range = list(range(D))
            
            k_corr = RBFKernel   (batch_shape=batch_shape, ard_num_dims=input_dims) #(Din,  active_dims=D_range[:Din], lengthscales=0.1,  variance=1.5, ARD=True)
            k_prev = RBFKernel   (batch_shape=batch_shape, ard_num_dims=input_dims) #(Dout, active_dims=D_range[Din:], variance = 1., lengthscales=0.1, ARD=True)
            k_bias = LinearKernel(power=1) # (Dout, active_dims=D_range[Din:], variance = 1e-6)
            k_in   = RBFKernel   (batch_shape=batch_shape, ard_num_dims=input_dims) #(Din,  active_dims=D_range[:Din], variance = 1e-6, lengthscales=1., ARD=True)
            
            k_l = k_corr*(k_prev + k_bias) + k_in
            kernels.append(k_l)

        # for i, kernel in enumerate(kernels[:-1]):
        #     kernels[i] += WhiteNoiseKernel(1, variance=0.)

    # this is the kernel function
    def forward(self, x1, x2, **params): # Concatenar los x y los f de las diferentes capas
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)

    # this is the kernel function
    def forward(self, x1, x2, **params):
        # apply lengthscale
        x1_ = x1.div(self.length)
        x2_ = x2.div(self.length)
        # calculate the distance between inputs
        diff = self.covar_dist(x1_, x2_, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        # return sinc(diff) = sin(diff) / diff
        return torch.sin(diff).div(diff)
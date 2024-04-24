import numpy as np
import torch

from enum import Enum

import gpytorch

import scipy.linalg as spla
from gpytorch.means.zero_mean import ZeroMean
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from mobocmf.util.util import triu_indices, compute_dist
from botorch.models.gp_regression import SingleTaskGP

class TL(Enum): # Type of lengthscale
    ONES = 1
    MEDIAN = 2
    CENTESIMAL = 3

class MFGP_lin(SingleTaskGP): # modos entrenar() y eval()

    # We assume the last column contains the fidelity

    def __init__(self, x_train, y_train, num_fidelities, type_lengthscale=TL.MEDIAN):

        self.input_dim = x_train.shape[ 1 ] - 1
        self.x_train = x_train
        self.y_train = y_train
        self.num_fidelities = num_fidelities

        mean_module = ZeroMean()
        covar_module = MFKernel_lin(x_train.shape[ 1 ], self.get_init_lengthscale(type_lengthscale, x_train[:, 0 : self.input_dim ]), num_fidelities)

        super().__init__(x_train, y_train, mean_module = mean_module, covar_module = covar_module)
        self.likelihood.noise = 1e-1
        self.double()

#    def forward(self, x):
#        mean_x = self.mean_module(x)
#        covar_x = self.covar_module(x)
#        return MultivariateNormal(mean_x, covar_x)

    def predict(self, x, fidelity):

        if len(x.shape) > 2:
            assert x.shape[ 1 ] == 1
            x = x[ :, 0, : ]

        self.eval()
        fidelity = fidelity * torch.ones((x.shape[ 0 ], 1)) 
        x = torch.concatenate([ x, fidelity ], 1)
        result = self(x)
        self.train()
        return result

    def get_init_lengthscale(self, type_lengthscale, inputs=None):

        dists_x_train = compute_dist(inputs)
        return torch.sqrt(torch.median(dists_x_train[ triu_indices(inputs.shape[ 0 ], 1) ]))

    # Returns a function with the posterior mean of the high fidelity to
    # use in optimization to estimate the pareto set.
    # Receives numpy arrays

    def get_mean_function_high_fidelity(self):

        def mean_function(x, gradient = False):

            if len(x.shape) != 2:
                x = x.reshape((1, len(x)))

            self.eval()

            x = np.concatenate([ x, (self.num_fidelities - 1) * np.ones((x.shape[ 0 ], 1)) ], 1)
            x = torch.from_numpy(x)

            if gradient == False:

                with torch.no_grad():
                    return self(x).mean.numpy()

            else:

                x_grad = torch.ones((x.shape[ 0 ], x.shape[ 1 ] - 1))

                for i in range(x.shape[ 0 ]):
                    x_input = x[ i : (i + 1), : ]
                    x_input = x_input.requires_grad_(True)
                    mean = self(x_input).mean
                    x_grad[ i, : ] = torch.autograd.grad(mean, x_input)[ 0 ][ 0, 0 : x_input.shape[ 1 ] - 1 ]

                return x_grad.numpy()

        return mean_function



class MFKernel_lin(gpytorch.kernels.Kernel):

    is_stationary = True

    def __init__(self, input_dim, init_lengthscale, num_fidelities, **kwargs):

        super().__init__(**kwargs)

        self.num_fidelities = num_fidelities
        self.input_dim = input_dim

        batch_shape = torch.Size([])
        D_range = list(range(input_dim))

        self.cov_funct_noise = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim - 1,  \
                active_dims=D_range[ 0 : (input_dim - 1) ]), batch_shape=batch_shape, ard_num_dims=None)

        self.cov_funct_signal = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim - 1,  \
                active_dims=D_range[ 0 : (input_dim - 1) ]), batch_shape=batch_shape, ard_num_dims=None)

        self.cov_funct_noise.initialize(outputscale = 0.1)  
        self.cov_funct_signal.initialize(outputscale = 1.0)  

        self.cov_funct_noise.base_kernel.initialize(lengthscale=init_lengthscale)
        self.cov_funct_signal.base_kernel.initialize(lengthscale=init_lengthscale)

        self.register_parameter(name="rho", parameter=torch.nn.Parameter(0.5 * torch.ones(num_fidelities - 1)))

    def forward(self, x1, x2, **params):

        if len(x1.shape) > 2:
            assert x1.shape[ 0 ] == 1 
            x1_original = x1
            x1 = x1[ 0, :, : ]
            expand = True
        else:
            expand = False

        if len(x2.shape) > 2:
            assert x2.shape[ 0 ] == 1 
            x2_original = x2
            x2 = x2[ 0, :, : ]
            expand = True
        else:
            expand = False

        # We start counting fidelities at 0

        fidelities1 = x1[ :, (self.input_dim - 1) : self.input_dim ] + 1
        fidelities2 = x2[ :, (self.input_dim - 1) : self.input_dim ] + 1

        fidelities1_tmp = fidelities1.tile((1, x2.shape[ 0 ])) 
        fidelities2_tmp = fidelities2.tile((1, x1.shape[ 0 ])) 

        min_fidelities = torch.minimum(fidelities1_tmp, fidelities2_tmp.T) 

        factor_signal = torch.ones((x1.shape[ 0 ], x2.shape[ 0 ]))
        factor_noise = torch.zeros((x1.shape[ 0 ], x2.shape[ 0 ]))
        factor_noise_final = torch.zeros((x1.shape[ 0 ], x2.shape[ 0 ]))

        cum_prods = torch.concatenate([ torch.ones(1), torch.cumprod(self.rho, 0) ], 0)
        cum_prods_x1 = torch.gather(cum_prods.tile((x1.shape[ 0 ], 1)), 1, fidelities1.type(torch.int64) - 1)
        cum_prods_x2 = torch.gather(cum_prods.tile((x2.shape[ 0 ], 1)), 1, fidelities2.type(torch.int64) - 1)
        factor_signal_final = torch.outer(cum_prods_x1.flatten(), cum_prods_x2.flatten())

#        for i in range(x1.shape[ 0 ]):
#            for j in range(x2.shape[ 0 ]):
#                if fidelities1[ i ] > 1:
#                    factor_signal[ i, j ] *= torch.prod(self.rho[ 0 : int(fidelities1[ i ] - 1) ])
#                if fidelities2[ j ] > 1:
#                    factor_signal[ i, j ] *= torch.prod(self.rho[ 0 : int(fidelities2[ j ] - 1) ])

#        for i in range(x1.shape[ 0 ]):
#            for j in range(x2.shape[ 0 ]):
#                if min_fidelities[ i, j ] >= 2:
#                    factor_noise[ i, j ] += 1
#
#                if min_fidelities[ i, j ] >= 3:
#                    factor_noise[ i, j ] += torch.sum(self.rho[ 1 : int(min_fidelities[ i, j ] - 1) ]**2)

        factor_noise_final[ min_fidelities >= 2 ] += 1

        for k in range(3, self.num_fidelities - 1): 
            factor_noise_final[ min_fidelities >= k ] += self.rho[ k - 2 ]**2

        if expand == True:
            return factor_signal_final[ None, :, : ] * self.cov_funct_signal(x1_original, x2_original) + factor_noise_final[ None, : , : ] * self.cov_funct_noise(x1_original, x2_original)
        else:
            return factor_signal_final * self.cov_funct_signal(x1, x2) + factor_noise_final * self.cov_funct_noise(x1, x2)



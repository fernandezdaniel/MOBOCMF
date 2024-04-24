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
from gpytorch.constraints import Interval


class TL(Enum): # Type of lengthscale
    ONES = 1
    MEDIAN = 2
    CENTESIMAL = 3

class MFGP(ExactGP, GPyTorchModel): # modos entrenar() y eval()

    # We assume the last column contains the fidelity

    def __init__(self, x_train, y_train, num_fidelities, type_lengthscale=TL.MEDIAN):

        self.input_dim = x_train.shape[ 1 ] - 1

        self.x_train = x_train
        self.y_train = y_train
        self.num_fidelities = num_fidelities
        likelihood = GaussianLikelihood()
        likelihood.noise = 1e-1

        super().__init__(x_train, y_train.squeeze(-1), likelihood)

        self.mean_module = ZeroMean()
        self.covar_module = MFKernel(x_train.shape[ 1 ], self.get_init_lengthscale(type_lengthscale, x_train[:, 0 : self.input_dim ]))

        self.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

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

    def _phi_rbf(self, x, W, b, alpha, nFeatures, gradient=False):
        if gradient:
            return -np.sqrt(2.0 * alpha / nFeatures) * np.sin(W @ x.T + b) * W

        return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)

    def _chol2inv(self, chol):
        return spla.cho_solve((chol, False), np.eye(chol.shape[ 0 ]))

    def _rff_sample_posterior_weights(self, y_data, Phi):

        randomness  = np.random.normal(loc=0., scale=1., size=Phi.shape[ 0 ])

        A = Phi @ Phi.T  + np.eye(Phi.shape[ 0 ]) * self.likelihood.noise.detach().numpy()
        chol_A_inv = spla.cholesky(A)
        A_inv = self._chol2inv(chol_A_inv)

        m = spla.cho_solve((chol_A_inv, False), Phi @ y_data) 
        return m + (randomness @ spla.cholesky(self.likelihood.noise.detach().numpy() * A_inv, lower=False)).T

    def sample_from_posterior(self, fidelity, nFeatures=500):

        x_data = self.x_train.numpy()
        y_data = self.y_train.numpy()

        lengthscale_signal = self.covar_module.cov_funct_signal.base_kernel.lengthscale.detach().numpy().flatten()
        lengthscale_noise = self.covar_module.cov_funct_noise.base_kernel.lengthscale.detach().numpy().flatten()
        alpha_signal = self.covar_module.cov_funct_signal.outputscale.detach().numpy().item()
        alpha_noise = self.covar_module.cov_funct_noise.outputscale.detach().numpy().item()

        W_noise   = np.random.normal(size=(nFeatures, self.input_dim)) / lengthscale_noise
        b_noise   = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

        W_signal = np.random.normal(size=(nFeatures, self.input_dim)) / lengthscale_signal
        b_signal = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

        x_data_without_fidelity = x_data[ :, 0 : self.input_dim ]
        fidelities_train = x_data[ :, self.input_dim ]

        Phi_noise = self._phi_rbf(x_data_without_fidelity, W_noise, b_noise, alpha_noise, nFeatures)
        Phi_signal = self._phi_rbf(x_data_without_fidelity, W_signal, b_signal, alpha_signal, nFeatures)
        
        mask = np.ones((nFeatures * (self.num_fidelities - 1), x_data_without_fidelity.shape[ 0 ]))

        for i in range(x_data_without_fidelity.shape[ 0 ]):
            mask[ 0 : int(nFeatures * (self.num_fidelities - fidelities_train[ i ] - 1)), i ] = 0

        Phi_noise = np.tile(Phi_noise, ((self.num_fidelities - 1, 1))) * mask

        Phi = np.concatenate((Phi_signal, Phi_noise), 0)

        theta = self._rff_sample_posterior_weights(y_data[:, 0], Phi)
        
        def wrapper(x, gradient=False):

            if x.ndim == 1:
                x = x[ None, : ]

            if gradient: # The gradient computation is not prepared for broadcasting
                assert x.shape[ 0 ] == 1

            Phi_noise = self._phi_rbf(x, W_noise, b_noise, alpha_noise, nFeatures, gradient=gradient)
            Phi_signal = self._phi_rbf(x, W_signal, b_signal, alpha_signal, nFeatures, gradient=gradient)

            mask = np.ones(nFeatures * (self.num_fidelities - 1))
            mask[ 0 : (nFeatures * (self.num_fidelities - fidelity - 1)) ] = 0

            Phi_noise = np.tile(Phi_noise, ((self.num_fidelities - 1, 1))) * np.outer(mask, np.ones(x.shape[ 0]))

            features = np.concatenate((Phi_signal, Phi_noise), 0)

            return theta @ features

        return wrapper


 
class MFKernel(gpytorch.kernels.Kernel):

    is_stationary = True

    def __init__(self, input_dim, init_lengthscale, **kwargs):

        super().__init__(**kwargs)

        self.input_dim = input_dim

        batch_shape = torch.Size([])
        D_range = list(range(input_dim))

        self.cov_funct_noise = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim - 1,  \
                active_dims=D_range[ 0 : (input_dim - 1) ], lengthscale_constraint = Interval(1e-3,1000)), \
                batch_shape=batch_shape, ard_num_dims=None, outputscale_constraint = Interval(1e-3, 100))

        self.cov_funct_signal = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dim - 1,  \
                active_dims=D_range[ 0 : (input_dim - 1) ], lengthscale_constraint = Interval(1e-3,1000)), \
                batch_shape=batch_shape, ard_num_dims=None, outputscale_constraint = Interval(1e-3, 100))

        self.cov_funct_noise.initialize(outputscale = 0.1)  
        self.cov_funct_signal.initialize(outputscale = 1.0)  

        self.cov_funct_noise.base_kernel.initialize(lengthscale=init_lengthscale)
        self.cov_funct_signal.base_kernel.initialize(lengthscale=init_lengthscale)

    def forward(self, x1, x2, **params):

        fidelities1 = x1[ :, (self.input_dim - 1) : self.input_dim ]
        fidelities2 = x2[ :, (self.input_dim - 1) : self.input_dim ]

        # We start counting fidelities at 0

        fidelities1 = fidelities1.tile((1, x2.shape[ 0 ])) + 1
        fidelities2 = fidelities2.tile((1, x1.shape[ 0 ])) + 1

        min_fidelities = torch.minimum(fidelities1, fidelities2.T) 

        return self.cov_funct_signal(x1, x2) + (min_fidelities - 1) * self.cov_funct_noise(x1, x2)



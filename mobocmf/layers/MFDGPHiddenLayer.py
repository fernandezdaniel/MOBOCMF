import numpy as np
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel
from gpytorch.variational import UnwhitenedVariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.lazy import LazyEvaluatedKernelTensor

class CovarianceMatrixMF(LazyEvaluatedKernelTensor):

    def add_jitter(self, jitter_val = 2e-6):
        return super(LazyEvaluatedKernelTensor, self).add_jitter(jitter_val)

class MFDGPHiddenLayer(DeepGPLayer):

    # input_dims is the dimensionality of the attribute vector x that is put into the layer 

    def __init__(self, num_layer, input_dims, inducing_points, inducing_values, num_fidelities, y_high_std = 1.0):

        self.num_layer = num_layer
        self.input_dims = input_dims
        num_inducing = inducing_points.shape[ 0 ]
        batch_shape = torch.Size([])

        # We initialize the covariance function depending on whether we are a first layer or not

        if num_layer == 0:

            covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims, \
                active_dims=list(range(input_dims))), batch_shape=batch_shape, ard_num_dims=None)
            
            covar_module.base_kernel.initialize(lengthscale = 1.0 * np.ones(input_dims))
            covar_module.initialize(outputscale = 1.0)

        else:

            # We use the fact that the output dim of a layer is always 1 in this model

            D_range = list(range(input_dims))
            
            k_x_1 = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims - 1,  \
                active_dims=D_range[ 0 : (input_dims - 1) ]), batch_shape=batch_shape, ard_num_dims=None)

            k_f = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=1, \
                active_dims=D_range[ (input_dims - 1) : input_dims ]), batch_shape=batch_shape, ard_num_dims=None)

            k_x_2 = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims - 1,  \
                active_dims=D_range[ 0 : (input_dims - 1) ]), batch_shape=batch_shape, ard_num_dims=None)

            k_lin = LinearKernel(batch_shape=batch_shape, active_dims=D_range[ (input_dims - 1) : input_dims ])

            k_x_1.base_kernel.initialize(lengthscale = 1.0 * np.ones(input_dims - 1))
            k_f.base_kernel.initialize(lengthscale = 1.0 * np.ones(1))
            k_x_2.base_kernel.initialize(lengthscale = 1.0 * np.ones(input_dims - 1))
            k_lin.initialize(variance = 1.0 * np.ones(1))

            k_x_1.initialize(outputscale = 1.0)
            k_f.initialize(outputscale = 1.0)
            k_x_2.initialize(outputscale = 1.0)

            covar_module = k_x_1 * (k_lin + k_f) + k_x_2

        # We initialize the variational approximation

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing, \
            batch_shape=batch_shape, mean_init_std = 0.0)

        # We initialize the covariance matrix to something diagonal with small variances. In the high fidelity we use the prior.

        if num_layer == num_fidelities - 1:
            init_dist = gpytorch.distributions.MultivariateNormal(inducing_values, \
                covar_module(inducing_points) * (1e-2 * y_high_std**2)**2)
        else:
            init_dist = gpytorch.distributions.MultivariateNormal(inducing_values, torch.eye(num_inducing) * 1e-8)

        variational_distribution.initialize_variational_distribution(init_dist)

        variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, \
            learn_inducing_locations=False)
        variational_strategy.variational_params_initialized = torch.tensor(1) # XXX DHL This avoids random initialization

        # XXX DHL None argument generates MultivNormal not MultiTaskMultivarnormal, used often at the last layer, 
        # but here we use it at all layers

        super(MFDGPHiddenLayer, self).__init__(variational_strategy, input_dims, None) 

        self.mean_module = ZeroMean()
        self.covar_module = covar_module

    def forward(self, x): 

        # We check that the input is the right one

        assert x.shape[ -1 ] == self.input_dims

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x.__class__ = CovarianceMatrixMF # This replaces the add_jitter method which uses 1e-3 by default.

        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):

        # This adds extra inputs, if given

        if len(other_inputs):

            if isinstance(x, gpytorch.distributions.MultivariateNormal):
                x = x.rsample()

            processed_inputs = []

            for inp in other_inputs:

                if isinstance(inp, gpytorch.distributions.MultivariateNormal):
                    inp = inp.rsample().T

                processed_inputs.append(inp)

            x = torch.cat([x] + processed_inputs, dim=-1)
        
        return super().__call__(x, are_samples=bool(len(other_inputs)))

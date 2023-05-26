import numpy as np
import scipy.linalg as spla
import torch
import gpytorch

from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel
from gpytorch.variational import UnwhitenedVariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.lazy import LazyEvaluatedKernelTensor

# from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.samplers import SobolQMCNormalSampler

class CovarianceMatrixMF(LazyEvaluatedKernelTensor):

    def add_jitter(self, jitter_val = 2e-6):
        return super(LazyEvaluatedKernelTensor, self).add_jitter(jitter_val)

# necesita el modo_entrenar() y eval(), en el segundo modo llamamos a MC.
class MFDGPHiddenLayer(DeepGPLayer): # Lo fijo son las muestras de la Gausiana estandar y especificas para cada capa y reusar # Hacer un metodo que sea fixed_sample(x <- del cual generar la muestra) (num_sample, self.num_layer) con samples fijos para la capa, solo tiene que generar un sample

    # input_dims is the dimensionality of the attribute vector x that is put into the layer 

    def __init__(self, num_layer, input_dims, inducing_points, inducing_values, num_fidelities, y_high_std=1.0, num_samples=10):
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

            self.sample_from_posterior = self._sample_from_posterior_layer0
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

            self.sample_from_posterior = self._sample_from_posterior

        # We initialize the variational approximation

        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing,
                                                                   batch_shape=batch_shape,
                                                                   mean_init_std=0.0)

        # We initialize the covariance matrix to something diagonal with small variances. In the high fidelity we use the prior.

        if num_layer == num_fidelities - 1:
            init_dist = MultivariateNormal(inducing_values,
                                           covar_module(inducing_points) * (1e-2 * y_high_std**2)**2)
        else:
            init_dist = MultivariateNormal(inducing_values, torch.eye(num_inducing) * 1e-8)

        variational_distribution.initialize_variational_distribution(init_dist)

        variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution,
                                                             learn_inducing_locations=False)
        variational_strategy.variational_params_initialized = torch.tensor(1) # XXX DHL This avoids random initialization

        # XXX DHL None argument generates MultivNormal not MultiTaskMultivarnormal, used often at the last layer, 
        # but here we use it at all layers

        super(MFDGPHiddenLayer, self).__init__(variational_strategy, input_dims, None) 

        self.mean_module = ZeroMean()
        self.covar_module = covar_module

        # DFS Parameters for QMC-sampling

        # samples = other_inputs[0].get_base_samples(sample_shape=torch.Size([10])); inp_.rsample(base_samples=samples)
        # self.sampler = SobolQMCNormalSampler(num_samples=num_QMC_samples, seed=seed_QMC_samples)
        
        # self.base_samples_other_inp = get_base_samples(sample_shape=torch.Size([num_QMC_samples]))
        self.samples = 1.0 #torch.normal(mean=torch.zeros([num_samples]), std=torch.ones([num_samples]))[ : , None ]
        self.num_samples = num_samples
        self._eval_mode = False


    def train_mode(self):
        self._eval_mode = False
    
    def eval_mode(self):
        self._eval_mode = True

    def forward(self, x): 

        # We check that the input is the right one

        assert x.shape[ -1 ] == self.input_dims

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x.__class__ = CovarianceMatrixMF # This replaces the add_jitter method which uses 2e-6 by default.

        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs): # Cambiarlo para usar muestra fijas!! (10 muestras que sean siempre las mismas) Problema: deben aprox la distrib

        # This adds extra inputs, if given

        if len(other_inputs): # Layer different of the firstone

            if isinstance(x, gpytorch.distributions.MultivariateNormal):
                raise ValueError
                x = x.rsample()

            processed_inputs = []

            for inp in other_inputs:

                if isinstance(inp, gpytorch.distributions.MultivariateNormal):
                    # inp = inp.rsample().T
                    if self._eval_mode:
                        inp = torch.reshape(inp.mean + torch.sqrt(inp.variance) * self.samples, x.shape)
                    else:
                        inp = inp.rsample().T

                processed_inputs.append(inp)

            x = torch.cat([x] + processed_inputs, dim=-1)
        
        return super().__call__(x, are_samples=bool(len(other_inputs)))

    def _phi_rbf(self, x, W, b, alpha, nFeatures, gradient=False):
        if gradient:
            return -np.sqrt(2.0 * alpha / nFeatures) * np.sin(W @ x.T + b) * W

        return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)

    def _chol2inv(self, chol):
        return spla.cho_solve((chol, False), np.eye(chol.shape[ 0 ]))

    def _rff_sample_posterior_weights(self, y_data, S, Phi, sigma2=1e-6):

        randomness  = np.random.normal(loc=0., scale=1., size=Phi.shape[ 0 ])

        A = Phi @ Phi.T + sigma2 * np.eye(Phi.shape[ 0 ])
        chol_A_inv = spla.cholesky(A)
        A_inv = self._chol2inv(chol_A_inv)

        m = spla.cho_solve((chol_A_inv, False), Phi @ y_data)
        extraVar = (A_inv @ Phi) @ S @ (Phi.T @ A_inv)
        return m + (randomness @ spla.cholesky(sigma2 * A_inv + extraVar, lower=False)).T

    def _sample_from_posterior_layer0(self, input_dim, sample_from_posterior_last_layer=None, nFeatures=500):

        assert sample_from_posterior_last_layer is None

        x_data = self.variational_strategy.inducing_points.detach().numpy()
        y_data = self.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]
        S_data = self.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()

        lengthscale = self.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
        alpha  = self.covar_module.outputscale.detach().numpy().item()

        W   = np.random.normal(size=(nFeatures, input_dim)) / lengthscale
        b   = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))
        Phi = self._phi_rbf(x_data, W, b, alpha, nFeatures)
        
        theta = self._rff_sample_posterior_weights(y_data[:, 0], S_data, Phi)
        
        def wrapper(x, gradient=False):
            if gradient: # The gradient computation is not prepared for broadcasting
                assert x.shape[ 0 ] == 1

            if x.ndim == 1:
                x = x[ None, : ]

            features = self._phi_rbf(x, W, b, alpha, nFeatures, gradient=gradient)
            return theta @ features

        return wrapper
    
    def _sample_from_posterior(self, input_dim, sample_from_posterior_last_layer=None, nFeatures=500):

        assert sample_from_posterior_last_layer is not None

        xf_data = self.variational_strategy.inducing_points.detach().numpy()
        x_data = xf_data[:, 0 : (xf_data.shape[ 1 ] - 1) ]
        f_data = xf_data[:, xf_data.shape[ 1 ] - 1 ]
        y_data = self.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]
        S_data = self.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()

        lengthscale_x1 = self.covar_module.kernels[ 0 ].kernels[ 0 ].base_kernel.lengthscale.detach().numpy().flatten()
        lengthscale_f  = self.covar_module.kernels[ 0 ].kernels[ 1 ].kernels[ 1 ].base_kernel.lengthscale.detach().numpy().flatten()
        lengthscale_x2 = self.covar_module.kernels[ 1 ].base_kernel.lengthscale.detach().numpy().flatten()

        alpha_x1  = self.covar_module.kernels[ 0 ].kernels[ 0 ].outputscale.detach().numpy().item()
        alpha_f   = self.covar_module.kernels[ 0 ].kernels[ 1 ].kernels[ 1 ].outputscale.detach().numpy().item()
        alpha_x1f = alpha_x1 * alpha_f
        alpha_x2  = self.covar_module.kernels[ 1 ].outputscale.detach().numpy().item()

        nu_lin    = self.covar_module.kernels[ 0 ].kernels[ 1 ].kernels[ 0 ].variance.detach().numpy().item()

        W_x1  = np.random.normal(size=(nFeatures, input_dim)) / lengthscale_x1
        W_f   = np.random.normal(size=nFeatures) / lengthscale_f
        W_x1f = np.concatenate([W_x1, W_f[ : , None]], axis=1)
        W_x2  = np.random.normal(size=(nFeatures, input_dim)) / lengthscale_x2

        b_x1  = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))
        b_x1f = b_x1
        b_x2  = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

        Phi_x1  = self._phi_rbf(x_data,  W_x1,  b_x1,  alpha_x1, nFeatures)
        Phi_x1f = self._phi_rbf(xf_data, W_x1f, b_x1f, alpha_x1f, nFeatures)
        Phi_x2  = self._phi_rbf(x_data,  W_x2,  b_x2,  alpha_x2, nFeatures)

        Phi = np.concatenate([(Phi_x1 * f_data * np.sqrt(nu_lin)), Phi_x1f, Phi_x2])

        theta = self._rff_sample_posterior_weights(y_data[:, 0], S_data, Phi)

        def wrapper(x, gradient=False):

            if x.ndim == 1:
                x = x[ None, : ]

            if gradient:
                
                assert x.shape[ 0 ] == 1
                
                f = sample_from_posterior_last_layer(x)
                xf = np.concatenate([x, f[ : , None]], axis=1)

                features_x1  = self._phi_rbf(x,  W_x1,  b_x1,  alpha_x1, nFeatures, gradient=False)
                features_x1f = self._phi_rbf(xf, W_x1f, b_x1f, alpha_x1f, nFeatures, gradient=False)
                features_x2  = self._phi_rbf(x,  W_x2,  b_x2,  alpha_x2, nFeatures, gradient=False)

                df_dx = sample_from_posterior_last_layer(x, gradient=True)

                dfeatures_xf_dx = np.concatenate([np.eye(x.shape[ 1 ]), df_dx[:,None]], axis=1)

                dfeatures_x1_dx   = self._phi_rbf(x,  W_x1,  b_x1,  alpha_x1, nFeatures, gradient=True)
                dfeatures_x1f_dxf = self._phi_rbf(xf, W_x1f, b_x1f, alpha_x1f, nFeatures, gradient=True)
                dfeatures_x2_dx   = self._phi_rbf(x,  W_x2,  b_x2,  alpha_x2, nFeatures, gradient=True)

                dfeatures_x1f_dx = dfeatures_x1f_dxf @ dfeatures_xf_dx.T

                features = np.concatenate([((dfeatures_x1_dx * f + df_dx * features_x1) * np.sqrt(nu_lin)),
                                             dfeatures_x1f_dx,
                                             dfeatures_x2_dx])
            else:
                
                f = sample_from_posterior_last_layer(x)
                xf = np.concatenate([x, f[ : , None]], axis=1)

                features_x1  = self._phi_rbf(x,  W_x1,  b_x1,  alpha_x1, nFeatures, gradient=False)
                features_x1f = self._phi_rbf(xf, W_x1f, b_x1f, alpha_x1f, nFeatures, gradient=False)
                features_x2  = self._phi_rbf(x,  W_x2,  b_x2,  alpha_x2, nFeatures, gradient=False)

                features = np.concatenate([(features_x1 * f * np.sqrt(nu_lin)), features_x1f, features_x2])
                
            return theta @ features

        return wrapper

import numpy as np
import torch

from enum import Enum

from gpytorch.models.deep_gps import DeepGP
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood # Usar FixedNoiseGaussianLikelihood para incljuir los puntos de la frontera
from gpytorch.constraints import GreaterThan, Interval

from mobocmf.layers.mfdgp_hidden_layer import MFDGPHiddenLayer
from mobocmf.util.util import triu_indices, compute_dist

class TL(Enum): # Type of lengthscale
    ONES = 1
    MEDIAN = 2
    CENTESIMAL = 3

class MFDGP(DeepGP): # modos entrenar() y eval()

    def __init__(self, x_train, y_train, fidelities, num_fidelities, num_samples=30, type_lengthscale=TL.MEDIAN):

        hidden_layers = []

        self._eval_mode = False
        self.num_samples = num_samples

        self.input_dims = x_train.shape[ -1 ]
        y_high_std = np.std(y_train[ (fidelities == num_fidelities - 1).flatten() ].numpy())

        # We set the inducing_points to the observations in the first layer

        to_sel = (fidelities == 0).flatten()
        inducing_points = x_train[ to_sel, : ]
        inducing_values = y_train[ to_sel, : ].flatten()
        init_lengthscale = self.get_init_lengthscale(type_lengthscale, inputs=inducing_points)
        hidden_layers.append(MFDGPHiddenLayer(input_dims=self.input_dims,
                                              num_layer=0,
                                              inducing_points=inducing_points,
                                              inducing_values=inducing_values,
                                              init_lengthscale=init_lengthscale,
                                              num_fidelities=num_fidelities,
                                              num_samples=0))

        for i in range(1, num_fidelities):

            # We set the inducing_points to the observations of the previous fidelity in later layers. We take only odd observatios!!!

            to_sel = (fidelities == i - 1).flatten()
            
            # inducing_points = torch.cat((x_train[ to_sel, : ][::2], y_train[ to_sel, : ][::2]), 1)
            # inducing_values = y_train[ to_sel, : ][::2].flatten() * 0.0
            inducing_points = torch.cat((x_train[ to_sel, : ], y_train[ to_sel, : ]), 1)
            inducing_values = y_train[ to_sel, : ].flatten() #* 0.0

            # fid_sel = (fidelities == i).flatten()
            # inducing_values = self.clip_inducing_values(x_train[ to_sel, : ], x_train[ fid_sel, : ], y_train[ fid_sel, : ].flatten())

            init_lengthscale = self.get_init_lengthscale(type_lengthscale, inputs=inducing_points)
            hidden_layers.append(MFDGPHiddenLayer(input_dims=self.input_dims + 1,
                                                  num_layer=i,
                                                  inducing_points=inducing_points,
                                                  inducing_values=inducing_values,
                                                  num_fidelities=num_fidelities,
                                                  init_lengthscale=init_lengthscale,
                                                  y_high_std=y_high_std,
                                                  num_samples=num_samples))

        super().__init__()

        self.name_hidden_layer = "hidden_layer_"
        self.name_hidden_layer_likelihood = "hidden_layer_likelihood_"
        self.name_hidden_layer_likelihood_noiseless = "hidden_layer_likelihood_noiseless_"
        self.num_hidden_layers = num_fidelities
        self.num_fidelities = num_fidelities

        # We add as many likelihoods as as layers (important since the noises can be different for each fidelity)

        for i, hidden_layer in enumerate(hidden_layers):
            setattr(self, self.name_hidden_layer + str(i), hidden_layer)
            likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-8, upper_bound=0.1*y_high_std))

            # We add a noiseless likelhood (with constraints) for conditional training. This will account for noiseless observations.
                
            if i == self.num_fidelities - 1:
                likelihood.noise = 1e-2 * y_high_std
            else:
                likelihood.noise = 1e-6     # We assume low noise initially

            setattr(self, self.name_hidden_layer_likelihood + str(i), likelihood)
    
    def clip_inducing_values(self, x_0, x_1, y_1):

        # Compute distances
        distances = torch.cdist(x_0, x_1)

        # find closest location between points
        indices_min = torch.argmin(distances, dim=1)

        return y_1[ indices_min ]

    def get_init_lengthscale(self, type_lengthscale, inputs=None):

        if type_lengthscale == TL.ONES:
            return torch.ones(self.input_dims)

        elif type_lengthscale == TL.MEDIAN:
            dists_x_train = compute_dist(inputs)
            return torch.sqrt(torch.median(dists_x_train[ triu_indices(inputs.shape[ 0 ], 1) ]))
            # return 0.5 * torch.log(torch.median(dists_x_train[ triu_indices(inputs.shape[ 0 ], 1) ]))

        elif type_lengthscale == TL.CENTESIMAL:
            return 0.01 * np.ones(self.input_dims)
        
        else:
            ValueError("Wrong type of lengthscale.")

    def train_mode(self):

        for i in range(self.num_hidden_layers):

            hidden_layer = getattr(self, self.name_hidden_layer + str(i))

            hidden_layer.train_mode()

        self._eval_mode = False

    def eval_mode(self):

        for i in range(self.num_hidden_layers):

            hidden_layer = getattr(self, self.name_hidden_layer + str(i))

            hidden_layer.eval_mode()

        self._eval_mode = True


    def forward(self, inputs, max_fidelity=None):

        num_layers = self.num_hidden_layers if max_fidelity is None else max_fidelity + 1

        # We propagate data through the layers and return all layers outputs

        l_outputs = [ torch.full(inputs.shape, torch.nan) ] * num_layers

        for i in range(num_layers):

            hidden_layer = getattr(self, self.name_hidden_layer + str(i))

            if i == 0:
                output_layer = hidden_layer(inputs)
            else:
                output_layer = hidden_layer(inputs, output_layer)

            l_outputs[ i ] = output_layer

        return l_outputs

    def fix_variational_hypers(self, value):

        for i in range(self.num_hidden_layers):
            likelihood = getattr(self, self.name_hidden_layer_likelihood + str(i))
            likelihood.raw_noise.requires_grad = not value
       
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i))
            hidden_layer.variational_strategy._variational_distribution.chol_variational_covar.requires_grad = not value
            
    def fix_variational_hypers_cond(self, value): # Cambiar para que se fijen todos los parametros excepto la media y la var del modelo

        for i in range(self.num_hidden_layers):
            likelihood = getattr(self, self.name_hidden_layer_likelihood + str(i))
            likelihood.raw_noise.requires_grad = not value
       
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i))

            for (name, param) in hidden_layer.covar_module.named_parameters():
                param.requires_grad = not value

    def predict(self, test_x, fidelity_layer=0):

        # Computes a sample from the predictive distribution calling propagate in the model
        # Note: in a DeepGP call calls eventually the forward method. See gpytorch code.

        assert fidelity_layer >= 0 and fidelity_layer < self.num_fidelities

        with torch.no_grad():
            mus = []
            variances = []

            likelihood = getattr(self, self.name_hidden_layer_likelihood + str(fidelity_layer))
            preds = likelihood(self(test_x, max_fidelity=fidelity_layer)[ fidelity_layer ]) # DFS: Changed, ask  DHL before: likelihood(self(test_x)[ fidelity_layer ])
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
    
    def predict_for_acquisition(self, test_x, fidelity_layer=0): # Que devuelva de todas  las fidelities (Eficiente, pasa los datos solo unaa vez por capa)

        # Computes a sample from the predictive distribution calling propagate in the model
        # Note: in a DeepGP call calls eventually the forward method. See gpytorch code.

        # test_x = torch.tile(test_x, (2, 1, 1))

        # assert test_x.shape[ 0 ] == 1

        # test_x_tile = torch.tile(test_x, (self.num_samples, 1))
        
        test_x_tile = torch.tile(test_x, (self.num_samples, 1, 1))

        self.eval_mode()

        mus_tilde, vars_tilde = self.predict(test_x_tile, fidelity_layer=fidelity_layer)

        self.train_mode()

        mus  = torch.mean(torch.reshape(mus_tilde, (self.num_samples, test_x.shape[ 0 ], test_x.shape[ 1 ])), 0)
        second_moment = torch.mean(torch.reshape(vars_tilde + mus_tilde**2, (self.num_samples, test_x.shape[ 0 ], test_x.shape[ 1 ])), 0)
        vars = second_moment - mus**2

        return mus.T, vars.T

    def sample_function_from_each_layer(self):

        result = []
        sample_from_posterior_last_layer = None
        
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i))
            sample = hidden_layer.sample_from_posterior(self.input_dims, sample_from_posterior_last_layer)
            sample_from_posterior_last_layer = sample
            result.append(sample)

        return result


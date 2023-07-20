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

    def __init__(self, x_train, y_train, fidelities, num_fidelities, type_lengthscale=TL.MEDIAN, num_samples_for_acquisition = 25, previously_trained_model = None):

        hidden_layers = []

        self._eval_mode = False
        self.num_samples_for_acquisition = num_samples_for_acquisition

        self.input_dims = x_train.shape[ -1 ]
        y_high_std = np.std(y_train[ (fidelities == num_fidelities - 1).flatten() ].numpy())

        # We check if we are given a previously_trained_model

        if previously_trained_model is not None:
            previously_trained_layer = getattr(previously_trained_model, previously_trained_model.name_hidden_layer + str(0))
        else:
            previously_trained_layer = None

        # We set the inducing_points to the observations in the first layer

#        to_sel = (fidelities == 0).flatten()
#        inducing_points = x_train[ to_sel, : ]
#        inducing_values = y_train[ to_sel, : ].flatten()

        inducing_points_0, inducing_values_0 = self.find_good_initial_inducing_points_and_values(x_train, y_train, fidelities, 0)

        init_lengthscale = self.get_init_lengthscale(type_lengthscale, inputs=inducing_points_0)
        hidden_layers.append(MFDGPHiddenLayer(input_dims=self.input_dims,
                                              num_layer=0,
                                              inducing_points=inducing_points_0,
                                              inducing_values=inducing_values_0,
                                              init_lengthscale=init_lengthscale,
                                              num_fidelities=num_fidelities,
                                              num_samples_for_acquisition=num_samples_for_acquisition, 
                                              previously_trained_layer = previously_trained_layer))

        for i in range(1, num_fidelities):

            # We set the inducing_points to the observations of the previous fidelity in later layers. We take only odd observatios!!!

            #to_sel = (fidelities == i - 1).flatten()
            
            # inducing_points = torch.cat((x_train[ to_sel, : ][::2], y_train[ to_sel, : ][::2]), 1)
            # inducing_values = y_train[ to_sel, : ][::2].flatten() * 0.0
            #inducing_points = torch.cat((x_train[ to_sel, : ], y_train[ to_sel, : ]), 1)
            #inducing_values = y_train[ to_sel, : ].flatten() #* 0.0

            #inducing_points, inducing_values = self.find_good_initial_inducing_points_and_values(x_train, y_train, fidelities, i)
            inducing_points = torch.cat((inducing_points_0, inducing_values_0[:,None]), 1)
            inducing_values = inducing_values_0

            # fid_sel = (fidelities == i).flatten()
            # inducing_values = self.clip_inducing_values(x_train[ to_sel, : ], x_train[ fid_sel, : ], y_train[ fid_sel, : ].flatten())

            init_lengthscale = self.get_init_lengthscale(type_lengthscale, inputs=inducing_points)

            if previously_trained_model is not None:
                previously_trained_layer = getattr(previously_trained_model, previously_trained_model.name_hidden_layer + str(i))
            else:
                previously_trained_layer = None

            hidden_layers.append(MFDGPHiddenLayer(input_dims=self.input_dims + 1,
                                                  num_layer=i,
                                                  inducing_points=inducing_points,
                                                  inducing_values=inducing_values,
                                                  num_fidelities=num_fidelities,
                                                  init_lengthscale=init_lengthscale,
                                                  y_high_std=y_high_std,
                                                  num_samples_for_acquisition=num_samples_for_acquisition,
                                                  previously_trained_layer = previously_trained_layer))

        super().__init__()

        self.name_hidden_layer = "hidden_layer_"
        self.name_hidden_layer_likelihood = "hidden_layer_likelihood_"
        self.name_hidden_layer_likelihood_noiseless = "hidden_layer_likelihood_noiseless_"
        self.num_hidden_layers = num_fidelities
        self.num_fidelities = num_fidelities

        # We add as many likelihoods as as layers (important since the noises can be different for each fidelity)

        for i, hidden_layer in enumerate(hidden_layers):
            y_std = np.std(y_train[ (fidelities == i).flatten() ].numpy())
            setattr(self, self.name_hidden_layer + str(i), hidden_layer)
            likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-8, upper_bound=0.1*y_std))

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
#            hidden_layer.variational_strategy.inducing_points.requires_grad = not value
            
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

        mus = []
        variances = []

        likelihood = getattr(self, self.name_hidden_layer_likelihood + str(fidelity_layer))
        preds = likelihood(self(test_x, max_fidelity=fidelity_layer)[ fidelity_layer ]) # DFS: Changed, ask  DHL before: likelihood(self(test_x)[ fidelity_layer ])
        mus.append(preds.mean)
        variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
    
    def predict_for_acquisition(self, test_x, fidelity_layer=0): # Que devuelva de todas  las fidelities (Eficiente, pasa los datos solo una vez por capa)

        # We test if we have three indexes. Three indexes are given in optimize_acquisition by botorch.

        if len(test_x.shape) > 2:
            assert test_x.shape[ 1 ] == 1
            test_x = test_x[ :, 0, : ]

        # Computes a sample from the predictive distribution calling propagate in the model
        # Note: in a DeepGP call calls eventually the forward method. See gpytorch code.

        test_x_tile = test_x.repeat_interleave(self.num_samples_for_acquisition, 0)

        self.eval_mode()

        mus_tilde, vars_tilde = self.predict(test_x_tile, fidelity_layer=fidelity_layer)

        self.train_mode()

        # DHL: The reshape is done to undo the repeat_interleave above

        mus  = torch.mean(torch.reshape(mus_tilde, (test_x.shape[ 0 ], self.num_samples_for_acquisition)), 1)
        second_moment = torch.mean(torch.reshape(vars_tilde + mus_tilde**2, (test_x.shape[ 0 ], self.num_samples_for_acquisition)), 1)
        vars = second_moment - mus**2

        return mus, vars

    def sample_function_from_each_layer(self):

        result = []
        sample_from_posterior_last_layer = None
        
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i))
            sample = hidden_layer.sample_from_posterior(self.input_dims, sample_from_posterior_last_layer)
            sample_from_posterior_last_layer = sample
            result.append(sample)

        return result

    def sample_function_from_prior_each_layer(self):

        result = []
        sample_from_prior_last_layer = None
        
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i))
            sample = hidden_layer.sample_from_prior(self.input_dims, sample_from_prior_last_layer)
            sample_from_prior_last_layer = sample
            result.append(sample)

        return result

    def find_good_initial_inducing_points_and_values(self, x_train, y_train, fidelities, layer):

        num_inducing_points = 100

        if layer == 0:

            # We sample uniformly in the unit box the inducing points

            inducing_points = torch.rand(size = ((num_inducing_points, x_train.shape[ 1 ])))
            inducing_values = torch.ones(num_inducing_points)

            # We set the initial inducing values to the targets of the closest point for that fidelity
            
            for i in range(num_inducing_points):
                temporal_data = torch.cat((x_train[ fidelities[ :, 0 ] == 0, : ], inducing_points[ i : (i + 1), : ]), 0)
                to_sel = torch.argmin(compute_dist(temporal_data)[ 0 : (temporal_data.shape[ 0 ] - 1), temporal_data.shape[ 0 ] - 1])
                inducing_values[ i ] = y_train[ fidelities[ :, 0 ] == 0 , : ][ to_sel ]

        else:

            # We sample uniformly in the unit box the inducing points

            inducing_points = torch.rand(size = ((num_inducing_points, x_train.shape[ 1 ])))
            inducing_values = torch.ones(num_inducing_points)
            inducing_values_previous_layer = torch.ones(num_inducing_points)

            # We set the initial inducing values to the targets of the closest point for that fidelity
            
            for i in range(num_inducing_points):
                temporal_data = torch.cat((x_train[ fidelities[ :, 0 ] == layer, : ], inducing_points[ i : (i + 1), : ]), 0)
                to_sel = torch.argmin(compute_dist(temporal_data)[ 0 : (temporal_data.shape[ 0 ] - 1), temporal_data.shape[ 0 ] - 1])
                inducing_values[ i ] = y_train[ fidelities[ :, 0 ] == layer , : ][ to_sel ]

                # The output from the previous layer is set to the closest target value of the previous fidelity

                temporal_data = torch.cat((x_train[ fidelities[ :, 0 ] == layer - 1, : ], inducing_points[ i : (i + 1), : ]), 0)
                to_sel = torch.argmin(compute_dist(temporal_data)[ 0 : (temporal_data.shape[ 0 ] - 1), temporal_data.shape[ 0 ] - 1])
                inducing_values_previous_layer[ i ] = y_train[ fidelities[ :, 0 ] == layer - 1, : ][ to_sel ]

            inducing_points = torch.cat((inducing_points, inducing_values_previous_layer[ : , None ]), 1)

        return inducing_points, inducing_values





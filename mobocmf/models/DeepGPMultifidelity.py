import torch
from gpytorch.models.deep_gps import DeepGP
from gpytorch.likelihoods import GaussianLikelihood
from mobocmf.layers.MFDGPHiddenLayer import MFDGPHiddenLayer
from gpytorch.constraints import GreaterThan
import numpy as np

class DeepGPMultifidelity(DeepGP):

    def __init__(self, x_train, y_train, fidelities, num_fidelities):

        hidden_layers = []

        input_dims = x_train.shape[ -1 ]
        y_high_std = np.std(y_train[ fidelities == num_fidelities - 1 ].numpy())

        for i in range(num_fidelities):
            if i == 0:

                # We set the inducing_points to the observations in the first layer

                to_sel = (fidelities == 0).flatten()
                inducing_points = torch.from_numpy(x_train.numpy()[ to_sel, : ])
                inducing_values = torch.from_numpy(y_train.numpy()[ to_sel, : ].flatten())
                hidden_layers.append(MFDGPHiddenLayer(input_dims=input_dims, num_layer=i, inducing_points=inducing_points, \
                    inducing_values = inducing_values, num_fidelities = num_fidelities))
            else:

                # We set the inducing_points to the observations of the previous fidelity in later layers. We take odd observatios!!!

                to_sel = (fidelities == i - 1).flatten()
                inducing_points = torch.from_numpy(np.hstack((x_train.numpy()[ to_sel, : ][::2], (y_train.numpy()[ to_sel, : ][::2]))))
                inducing_values = torch.from_numpy(y_train.numpy()[ to_sel, : ][::2].flatten()) * 0.0
                hidden_layers.append(MFDGPHiddenLayer(input_dims=input_dims + 1, num_layer=i, \
                    inducing_points=inducing_points, inducing_values = inducing_values, num_fidelities = num_fidelities, \
                    y_high_std = np.std(y_train.numpy()[ fidelities == num_fidelities - 1 ])))

        super().__init__()

        self.name_hidden_layer = "hidden_layer_"
        self.name_hidden_layer_likelihood = "hidden_layer_likelihood_"
        self.num_hidden_layers = num_fidelities
        self.num_fidelities = num_fidelities

        # We add as many likelihoods as as layers (important since the noises can be different for each fidelity)

        for i, hidden_layer in enumerate(hidden_layers):
            setattr(self, self.name_hidden_layer + str(i), hidden_layer)
            likelihood = GaussianLikelihood(noise_constraint = GreaterThan(1e-8))

            if i == self.num_fidelities - 1:
                likelihood.noise = 1e-2 * y_high_std
            else:
                likelihood.noise = 1e-6     # We assume low noise initially

            setattr(self, self.name_hidden_layer_likelihood + str(i), likelihood)

    def forward(self, inputs):

        # We propagate data through the layers and return all layers outputs

        l_outputs = [ torch.full(inputs.shape, torch.nan) ] * self.num_hidden_layers

        for i in range(self.num_hidden_layers):

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

    def predict(self, test_x, fidelity_layer = 0):

        # Computes a sample from the predictive distribution calling propagate in the model
        # Note: in a DeepGP call calls eventually the forward method. See gpytorch code.

        assert fidelity_layer >= 0 and fidelity_layer < self.num_fidelities

        with torch.no_grad():
            mus = []
            variances = []
            likelihood = getattr(self, self.name_hidden_layer_likelihood + str(fidelity_layer))
            preds = likelihood(self(test_x)[ fidelity_layer ])
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)



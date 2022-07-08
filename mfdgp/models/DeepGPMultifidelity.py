import numpy as np
import torch
import tqdm
# from gpytorch.likelihoods import GaussianLikelihood
from likelihoods.GaussianLikelihoodMultifidelity import GaussianLikelihoodMultifidelity
from gpytorch.models.deep_gps import DeepGP

from layers.DeepGPMultifidelityHiddenLayer import DeepGPMultifidelityHiddenLayer

class DeepGPMultifidelity(DeepGP):
    def __init__(self, train_x_shape, num_hidden_layers, mean_type_hidden_layers='zero'):

        hidden_layers = []
        for i in range(num_hidden_layers):
            input_dims = train_x_shape[-1] if i == 0 else (1 + train_x_shape[-1])
            output_dims = 1 if i < num_hidden_layers - 1 else None # Is mandatory that the last layer has output_dims=None

            hidden_layers.append(DeepGPMultifidelityHiddenLayer(
                input_dims=input_dims,
                output_dims=output_dims,
                mean_type=mean_type_hidden_layers, #mean_type_hidden_layers if i < num_hidden_layers - 1 else 'constant',
                num_layer=i,
            ))

        super().__init__()

        self.name_hidden_layer = "hidden_layer_"
        self.num_hidden_layers = num_hidden_layers
        for i, hidden_layer in enumerate(hidden_layers):
            setattr(self, self.name_hidden_layer + str(i+1), hidden_layer)

        self.likelihood = GaussianLikelihoodMultifidelity()

    def propagate(self, inputs):

        l_outputs = [torch.full(inputs.shape, torch.nan)]*self.num_hidden_layers
        inputs_layer = inputs
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i+1))
            inputs_layer = hidden_layer(inputs_layer) if i == 0 else hidden_layer(inputs_layer, inputs) # Parece que falla aqui despues de haber puesto como que la media sea zero
            l_outputs[i] = inputs_layer

        return l_outputs

    def forward(self, inputs, all_fidelities=False, fidelity_layer=-1):
        if all_fidelities:
            return self.propagate(inputs)

        return self.propagate(inputs)[fidelity_layer]

    def predict(self, test_x, all_fidelities=False, fidelity_layer=-1):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            preds = self.likelihood(self(test_x, all_fidelities=all_fidelities, fidelity_layer=fidelity_layer))
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

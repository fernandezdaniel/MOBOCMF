import torch
from gpytorch.models.deep_gps import DeepGP
# from gpytorch.likelihoods import GaussianLikelihood

from mobocmf.likelihoods.GaussianLikelihoodMF import GaussianLikelihoodMF
from mobocmf.layers.MFDGPHiddenLayer import MFDGPHiddenLayer

class DeepGPMultifidelity(DeepGP):
    def __init__(self, shape_x_train, num_fidelities, mean_type_hidden_layers='constant'):

        hidden_layers = []
        for i in range(num_fidelities):
            input_dims = shape_x_train[-1] if i == 0 else (1 + shape_x_train[-1])
            output_dims = 1 if i < num_fidelities - 1 else None # Is mandatory that the last layer has output_dims=None

            hidden_layers.append(MFDGPHiddenLayer(
                input_dims=input_dims,
                output_dims=output_dims,
                mean_type=mean_type_hidden_layers, #mean_type_hidden_layers if i < num_hidden_layers - 1 else 'constant',
                num_layer=i,
            ))

        super().__init__()

        self.name_hidden_layer = "hidden_layer_"
        self.num_hidden_layers = num_fidelities
        for i, hidden_layer in enumerate(hidden_layers):
            setattr(self, self.name_hidden_layer + str(i+1), hidden_layer)

        self.likelihood = GaussianLikelihoodMF()

    def propagate(self, inputs):

        l_outputs = [torch.full(inputs.shape, torch.nan)]*self.num_hidden_layers
        inputs_layer = inputs
        for i in range(self.num_hidden_layers):
            hidden_layer = getattr(self, self.name_hidden_layer + str(i+1))
            inputs_layer = hidden_layer(inputs) if i == 0 else hidden_layer(inputs_layer, inputs) # DFS: It seems that it stops working here after setting the mean to zero
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
            # lls = []
            preds = self.likelihood(self(test_x, all_fidelities=all_fidelities, fidelity_layer=fidelity_layer))
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

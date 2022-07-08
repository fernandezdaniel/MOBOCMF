import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP

from botorch.models.gpytorch import GPyTorchModel

from layers.ToyDeepGPHiddenLayer import ToyDeepGPHiddenLayer

class MyDeepGP(DeepGP, GPyTorchModel):
    def __init__(self, train_x_shape, num_output_dims=1):
        hidden_layer_1 = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        # hidden_layer_2 = ToyDeepGPHiddenLayer(
        #     input_dims=hidden_layer_1.output_dims,
        #     output_dims=num_output_dims,
        #     mean_type='linear',
        # )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims, # input_dims=hidden_layer_2.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1
        # self.hidden_layer_2 = hidden_layer_2

        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

        self.double()

    def forward(self, inputs):  # Cambiar para tener en cuenta la nueva aquitectura que se utiliza en los problemas multifidelity
                                # Deberia calcular las predicciones para todas fidelidades dados unos inputs
        hidden_rep1 = self.hidden_layer_1(inputs)
        # hidden_rep2 = self.hidden_layer_2(hidden_rep1)

        output = self.last_layer(hidden_rep1) # (hidden_rep2)
        return output # Deberia devolver la distrib pred para cada fidelidad # Hay que tener en cuenta que para un batch no tenemos fidelidades (hemos observado unas pero otras no)

    def predict(self, test_x):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            preds = self.likelihood(self(test_x)) # Suma la varianza del likelihood
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)


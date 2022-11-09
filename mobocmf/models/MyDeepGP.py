import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from gpytorch.distributions import MultivariateNormal

from botorch.models.gpytorch import GPyTorchModel

from mobocmf.layers.DGPHiddenLayer import DGPHiddenLayer

class MyDeepGP(DeepGP, GPyTorchModel):
    def __init__(self, train_x_shape, num_output_dims=1):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

        self.double()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output, hidden_rep1 # Deberia devolver la distrib pred para cada fidelidad # Hay que tener en cuenta que para un batch no tenemos fidelidades (hemos observado unas pero otras no)

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            stddevs = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch)[0])
                mus.append(preds.mean)
                stddevs.append(preds.stddev)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)[0]))

        
        with torch.no_grad():
            mus0 = []
            stddevs0 = []
            lls0 = []
            for x_batch, y_batch in test_loader:
                _mt_mvn = self(x_batch)[1]
                mvn = MultivariateNormal(_mt_mvn.mean[0,...,0], _mt_mvn.covariance_matrix[0,:,:])
                preds0 = self.likelihood(mvn)
                mus0.append(preds0.mean)
                stddevs0.append(preds0.stddev)
                lls0.append(self.likelihood.log_marginal(y_batch, mvn))

        return torch.cat(mus, dim=-1), torch.cat(stddevs, dim=-1), torch.cat(lls, dim=-1), torch.cat(mus0, dim=-1), torch.cat(stddevs0, dim=-1), torch.cat(lls0, dim=-1)


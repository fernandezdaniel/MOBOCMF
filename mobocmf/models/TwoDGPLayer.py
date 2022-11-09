import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from mobocmf.layers.DGPHiddenLayer import DGPHiddenLayer

class TwoDGPLayer(DeepGP):
    def __init__(self, train_x_shape, lengthscale=None, covar_noise=None, fixed_noise=None, num_output_dims=1):

        assert not (covar_noise is not None and fixed_noise is not None)

        hidden_layer_1 = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )
            
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer_1.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        if lengthscale is not None:
            shape_tensor = hidden_layer_1.covar_module.base_kernel.raw_lengthscale.shape
            dtype_tensor = hidden_layer_1.covar_module.base_kernel.raw_lengthscale.dtype
            ini_lengthscale = torch.nn.Parameter(torch.tensor(lengthscale, dtype=dtype_tensor).reshape(shape_tensor))

            hidden_layer_1.covar_module.base_kernel.raw_lengthscale = ini_lengthscale
            last_layer.covar_module.base_kernel.raw_lengthscale     = ini_lengthscale

        super().__init__()

        self.hidden_layer_1 = hidden_layer_1

        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

        if covar_noise is not None:
            shape_tensor = self.likelihood.noise_covar.raw_noise.shape
            dtype_tensor = self.likelihood.noise_covar.raw_noise.dtype
            raw_noise = torch.nn.Parameter(torch.tensor(covar_noise, dtype=dtype_tensor).reshape(shape_tensor))
            self.likelihood.noise_covar.raw_noise = raw_noise

        if fixed_noise is not None:
            self.likelihood.noise = fixed_noise

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer_1(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():
            mus = []
            variances = []
            preds = self.likelihood(self(test_x))
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
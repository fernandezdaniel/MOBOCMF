import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer

class DeepGPMultifidelityHiddenLayer(DeepGPLayer):
    def __init__(self, num_layer, input_dims, output_dims, num_inducing=25, mean_type='constant'):

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPMultifidelityHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            self.mean_module = LinearMean(input_dims)
        elif mean_type == 'zero':
            self.mean_module = ZeroMean(input_dims)
        else:
            raise ValueError("mean_type " + mean_type + " is not recognized")

        Din  = input_dims - 1 if num_layer > 0 else input_dims
        Dout = output_dims if output_dims is not None else 1
        self.covar_module = \
        ScaleKernel(
             DeepGPMultifidelityHiddenLayer.make_mfdgp_kernel(num_layer, Din, Dout, batch_shape, lengthscale=0.05),
             batch_shape=batch_shape, ard_num_dims=None
        )

        # ScaleKernel(
        #     RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )


    # Given some inputs x, this function calculates the mean and variance of the predictive distribution and returns a multivariate gaussian of that mean and variances
    def forward(self, x): # Cambiar ligeramente de ser necesario para tener en cuenta la nueva aquitectura que se utiliza en los problemas multifidelity
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    # Given some inputs x this function returns the mean and variance at the end of the layer. Also this function allow us to add to the input of the layer new points 'other_inputs'
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

    @classmethod
    def make_mfdgp_kernel(cls, num_layer, Din, Dout, batch_shape, lengthscale=0.1, add_linear=True):
        
        if num_layer > 0:
            
            D = Din + Dout
            D_range = list(range(D))
            
            k_corr = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=Din,  active_dims=D_range[:Din], lengthscales=lengthscale, variance=1.0), batch_shape=batch_shape, ard_num_dims=None)
            k_prev = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=Dout, active_dims=D_range[Din:], lengthscales=lengthscale, variance=1.0), batch_shape=batch_shape, ard_num_dims=None)
            k_in   = ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=Din,  active_dims=D_range[:Din], lengthscales=lengthscale, variance=1.0), batch_shape=batch_shape, ard_num_dims=None)
            
            if add_linear:
                return k_corr * (k_prev + ScaleKernel(LinearKernel(batch_shape=batch_shape, ard_num_dims=Dout, active_dims=D_range[Din:], variance=1.0), batch_shape=batch_shape, ard_num_dims=None)) + k_in
            else:
                return k_corr * k_prev + k_in

        else:
            return ScaleKernel(RBFKernel(batch_shape=batch_shape, ard_num_dims=Din, active_dims=list(range(Din)), variance=1.0, lengthscales=lengthscale), batch_shape=batch_shape, ard_num_dims=None)
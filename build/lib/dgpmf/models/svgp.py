from torch import Tensor

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.containers import TrainingData

class SVGPModel(ApproximateGP, GPyTorchModel):

    def __init__(self, inducing_points: Tensor, kernel: Kernel = RBFKernel()):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPModel, self).__init__(variational_strategy) # super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        r"""Construct kwargs for the `SimpleCustomGP` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}

        
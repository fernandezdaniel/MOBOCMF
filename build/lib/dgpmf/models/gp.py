from torch import Tensor

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.containers import TrainingData

class ExactGPModel(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X: Tensor, train_Y: Tensor, likelihood: Likelihood = GaussianLikelihood(), kernel: Kernel = RBFKernel()):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

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
        
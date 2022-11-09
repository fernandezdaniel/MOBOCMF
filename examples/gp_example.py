from cProfile import label
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

class MyGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MyGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

LOW  = 0.0
HIGH = 1.0
# LOC_MAXIMUM = 0.5

NUM_FIDELITIES = 2

def foo(x):
    x = torch.from_numpy(x)

    return torch.sin(x * (2 * np.pi)).numpy()

def obj_mf1(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * obj_mf0(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd

def obj_mf0(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd

def train_gp(x_train, y_train, model, likelihood):
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    N_ITERS = 1000
    for i in range(N_ITERS):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, N_ITERS, loss.item()))
        optimizer.step()

    return loss.item(), model

def test_gp_example(seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    num_inputs = 8
    x_train = np.random.uniform(low=LOW, high=HIGH, size=(num_inputs, 1))
    x_train = np.sort(x_train)
    y_train_mf1 = obj_mf1(x_train)
    y_train_mf0 = obj_mf0(x_train)
    x_train = torch.from_numpy(x_train).float() * 10.0
    y_train_mf1 = torch.from_numpy(y_train_mf1[...,0]).float()
    y_train_mf0 = torch.from_numpy(y_train_mf0[...,0]).float()

    gp1_likelihood = GaussianLikelihood()
    GP1_model = MyGP(x_train, y_train_mf1, gp1_likelihood)
    loss_gp1, GP1_model = train_gp(x_train, y_train_mf1, GP1_model, gp1_likelihood)

    gp0_likelihood = GaussianLikelihood()
    GP0_model = MyGP(x_train, y_train_mf0, gp0_likelihood)
    loss_gp0, GP0_model = train_gp(x_train, y_train_mf0, GP0_model, gp0_likelihood)

    # We put the models into evaluation mode
    GP1_model.eval()
    gp1_likelihood.eval()
    GP0_model.eval()
    gp0_likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(LOW, HIGH*10.0, 201)
        gp1_observed_pred = gp1_likelihood(GP1_model(test_x))
        gp0_observed_pred = gp0_likelihood(GP0_model(test_x))

    with torch.no_grad():
        # Initialize plot
        _, ax = plt.subplots(1, 1, figsize=(4, 3))

        # We get upper and lower confidence bounds
        gp1_lower, gp1_upper = gp1_observed_pred.confidence_region()
        gp0_lower, gp0_upper = gp0_observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(x_train.numpy(), y_train_mf1.numpy(), 'k*')
        ax.plot(x_train.numpy(), y_train_mf0.numpy(), 'm*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), gp1_observed_pred.mean.numpy(), 'b--', label="GP_mf1")
        ax.fill_between(test_x.numpy(), gp1_lower.numpy(), gp1_upper.numpy(), alpha=0.5)
        ax.plot(test_x.numpy(), gp0_observed_pred.mean.numpy(), 'r--', label="GP_mf0")
        ax.fill_between(test_x.numpy(), gp0_lower.numpy(), gp0_upper.numpy(), alpha=0.5)

    ax.legend()

    plt.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":

    test_gp_example()

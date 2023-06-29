import torch
from gpytorch.mlls import VariationalELBO
import numpy as np

# Evaluates ELBO of the Multi-fidelity DGP

class VariationalELBOMF(VariationalELBO):

    # The constructor is the same as _ApproximateMarginalLogLikelihood, but the likelihood should be none since we have several
    # likelhoods, one per layer.

    # num_data should be the total number of points
    # num_fidelities should be the total number of fidelities

    def __init__(self, model, num_data, num_fidelities):

        super().__init__(None, model, num_data)
        self.num_data = num_data
        self.num_fidelities = num_fidelities

    # The forward method should receive the predictive distribution for the point, the target, and the corresponding fidelity
    # Call is a method that calls forward

    def forward(self, l_approximate_dist_f, target, fidelities, include_kl_term = True):

        assert target.shape[ 0 ] <= target.shape[ 1 ]   # This checks that the target has the proper shape

        num_batch = target.shape[ 1 ]
        data_term = 0.0

        for i in range(self.num_fidelities):

            if (fidelities == i).sum() != 0:
                likelihood = getattr(self.model, self.model.name_hidden_layer_likelihood + str(i))
                data_term += (likelihood.expected_log_prob(target, l_approximate_dist_f[ i ])[ fidelities.T == i ]).sum()

        if include_kl_term == False:
            return data_term

        kl_divergence = self.model.variational_strategy.kl_divergence()

        # We return elbo per batch

        return data_term - kl_divergence * num_batch / self.num_data, kl_divergence * num_batch / self.num_data


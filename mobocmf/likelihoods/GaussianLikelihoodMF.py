#!/usr/bin/env python3


import math
import torch
from torch import Tensor
from typing import Any

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood


class GaussianLikelihoodMF(GaussianLikelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    def expected_log_prob(self, target: Tensor, l_inputs: list, *params: Any, **kwargs: Any) -> Tensor:

        fidelity = 0
        mean     = l_inputs[fidelity].mean[...,0]     * (kwargs['l_fidelities'] == fidelity)[...,0]
        variance = l_inputs[fidelity].variance[...,0] * (kwargs['l_fidelities'] == fidelity)[...,0]
        if len(l_inputs) > 1:
            fidelity = 1
            for input in l_inputs[1:-1]:
                mean     += input.mean[...,0]     * (kwargs['l_fidelities'] == fidelity)[...,0]
                variance += input.variance[...,0] * (kwargs['l_fidelities'] == fidelity)[...,0]
                fidelity += 1
            mean     += l_inputs[fidelity].mean     * (kwargs['l_fidelities'] == fidelity)[...,0]
            variance += l_inputs[fidelity].variance * (kwargs['l_fidelities'] == fidelity)[...,0]
        
        num_event_dim = len(l_inputs[0].event_shape)

        noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *l_inputs[0].event_shape)

        res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)
        if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res

    
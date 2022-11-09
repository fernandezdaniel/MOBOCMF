#!/usr/bin/env python3

import torch
from gpytorch.mlls import VariationalELBO


class VariationalELBOMF(VariationalELBO):
    r"""
    The variational evidence lower bound (ELBO) for multifidelity.
    
    The matho expresion is pending...
    .. math::
       \begin{align*}
          \mathcal{L}_\text{ELBO-MF}
          \approx
          \sum_{t=1}^T \sum_{i=1}^N \mathbb{E}_{q( f_i^t)} \left[ \log p( y_i^t \! \mid \! f_i^t) \right]
          - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}
    """

    def forward(self, l_approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = l_approximate_dist_f[0].event_shape[0]
        log_likelihood = self._log_likelihood_term(l_approximate_dist_f, target, **kwargs).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior

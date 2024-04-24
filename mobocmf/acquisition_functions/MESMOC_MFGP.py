
import copy
import numpy as np

import torch
import gpytorch

from torch import Tensor

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch.distributions import Normal

from mobocmf.models.mfgp import MFGP

from botorch.optim.optimize import optimize_acqf # XXX delete choser part

CLAMP_LB = torch.finfo(torch.float32).eps

class _MES_MFGP(AnalyticAcquisitionFunction):
    def __init__(
        self,
        fidelity: int,
        model: MFGP,
        best_value:  float,
        is_constraint: bool
    ) -> None:
        
        super(AnalyticAcquisitionFunction, self).__init__(None)
        
        self.fidelity = fidelity
        self.is_constraint = is_constraint
        self.best_value = best_value
        self.model = model 

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate MES_MFGP
        """

        pred = self.model.predict(X, self.fidelity)
        pred_mean = pred.mean
        pred_variance = pred.variance

        # We compute moments of a truncated Gaussian and then add the noise

        noiseless_var = pred_variance
        stdv = noiseless_var.sqrt()
        normal = Normal(torch.zeros(1, device=X.device, dtype=X.dtype), torch.ones(1, device=X.device, dtype=X.dtype),)

        normalized_mvs = (self.best_value - pred_mean) / stdv
        cdf_mvs = normal.cdf(normalized_mvs).clamp_max(1-CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))

        if self.is_constraint == False:

            ratio = pdf_mvs / (1.0 - cdf_mvs)
            var_truncated = noiseless_var * (1 + (normalized_mvs - ratio) * ratio).clamp_min(CLAMP_LB)
            var_truncated_final = var_truncated + self.model.likelihood.noise
            entropy_cond = 0.5 * torch.log(var_truncated_final)
            entropy_uncond = 0.5 * torch.log(pred_variance + self.model.likelihood.noise)

            return torch.clamp(entropy_uncond - entropy_cond, min=0.0)
 
        else:

            # Best value is constraint threshold. We return the prob that it is larger
            # than that value (i.e. the constraint is feasible)
            
            return 1 - normal.cdf(normalized_mvs)

       

class MESMOC_MFGP():
    def __init__(
        self,
        objectives: dict,
        constraints: dict,
        input_dim: int,
        num_fidelities: int,
        best_objective_values: dict,
        constraint_thresholds: dict,
        standard_bounds = None,
    ) -> None:
        
        self.standard_bounds = standard_bounds
        self.num_fidelities = num_fidelities
        self.input_dim = input_dim

        self.objectives = objectives
        self.constraints = constraints
        self.costs_blackboxes = {}
        self.best_objective_values = best_objective_values
        self.constraint_thresholds = constraint_thresholds
        self.acquisition_objs = {}
        self.acquisition_cons = {}

        for n_f in range(0, num_fidelities):
            self.acquisition_objs[ n_f ] = {}
            self.acquisition_cons[ n_f ] = {}
            self.costs_blackboxes[ n_f ] = {}            
            self.costs_blackboxes[ n_f ][ "total" ] = 0.0

    def add_blackbox(self, fidelity: int, blackbox_name: str, cost_evaluation: float = 1.0, is_constraint=False):
            
        if is_constraint == False:
            jes_mfgp = _MES_MFGP(fidelity, self.objectives[ blackbox_name ], self.best_objective_values[ blackbox_name ], False)
            self.acquisition_objs[ fidelity ][ blackbox_name ] = jes_mfgp
            self.costs_blackboxes[ fidelity ][ "total" ] += cost_evaluation
            self.costs_blackboxes[ fidelity ][ blackbox_name ] = cost_evaluation
        else:
            jes_mfgp = _MES_MFGP(fidelity, self.constraints[ blackbox_name ], self.constraint_thresholds[ blackbox_name ], True)
            self.acquisition_cons[ fidelity ][ blackbox_name ] = jes_mfgp

        return jes_mfgp
    
    def coupled_acq(self, X: Tensor, fidelity: int) -> Tensor:

        acq_value = torch.zeros(size=(X.shape[ 0 ],))

        for name, acq in self.acquisition_objs[ fidelity ].items():
            acq_value += acq(X.double())

        prob_feasible = torch.ones(size=(X.shape[ 0 ],))

        for name, acq in self.acquisition_cons[ self.num_fidelities - 1 ].items():
            prob_feasible *= acq(X.double())

        return acq_value * prob_feasible

    def get_nextpoint_coupled(self, iteration=None, verbose=False) -> Tensor:

        if verbose: assert (iteration is not None)

        current_value_weighted = 0.0
        
        for fidelity in range(self.num_fidelities):

            new_candidate, new_values = optimize_acqf(acq_function=lambda x: self.coupled_acq(x, fidelity=fidelity), bounds=self.standard_bounds,
                q=1, num_restarts=5, raw_samples=200,  options={"maxiter": 200})
            
            new_values_weighted = new_values / self.costs_blackboxes[ fidelity ][ "total" ]
            
            if (fidelity == 0) or (current_value_weighted < new_values_weighted):

                fidelity_to_evaluate = fidelity
                current_value_weighted = new_values_weighted
                current_candidate = new_candidate

        nextpoint = current_candidate[ 0, : ]

        if verbose: 
            print("Iter:", iteration, "Acquisition: " + str(current_value_weighted.numpy() * \
                self.costs_blackboxes[ fidelity_to_evaluate ][ "total" ]) + " Evaluating fidelity", fidelity_to_evaluate, "at", nextpoint.numpy())

        return nextpoint, fidelity_to_evaluate

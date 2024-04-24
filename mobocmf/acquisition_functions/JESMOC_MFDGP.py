
import copy
import numpy as np

import torch
import gpytorch

from torch import Tensor

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

from mobocmf.util.blackbox_mfdgp_fitter import BlackBoxMFDGPFitter
from mobocmf.models.mfdgp import MFDGP

from botorch.optim.optimize import optimize_acqf # XXX delete choser part

class _JES_MFDGP(AnalyticAcquisitionFunction):
    def __init__(
        self,
        fidelity: int,
        mfdgp_uncond: MFDGP,
        mfdgp_cond: MFDGP,
        model: Model = None,
    ) -> None:
        
        assert model is None
        
        super(AnalyticAcquisitionFunction, self).__init__(model)
        
        self.fidelity = fidelity

        self.mfdgp_uncond = mfdgp_uncond
        self.mfdgp_cond = mfdgp_cond

    #@t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate JES_FMDGP
        """

        self.mfdgp_uncond.eval()
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_uncond, pred_variances_uncond = self.mfdgp_uncond.predict_for_acquisition(X, self.fidelity)
        self.mfdgp_uncond.train()

        self.mfdgp_cond.eval()
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_cond, pred_variances_cond = self.mfdgp_cond.predict_for_acquisition(X, self.fidelity)
        self.mfdgp_cond.train()

        return 0.5 * torch.clamp(torch.log(pred_variances_uncond) - torch.log(pred_variances_cond), min=0.0) 
        #return torch.clamp(pred_variances_uncond - pred_variances_cond, min=0.0)
        

class JESMOC_MFDGP():
    def __init__(
        self,
        model: BlackBoxMFDGPFitter,
        num_fidelities: int = 1,
        model_cond: BlackBoxMFDGPFitter = None,
        standard_bounds = None,
        eval_highest_fidelity: bool = False,
    ) -> None:
        
        self.standard_bounds = standard_bounds

        self.eval_highest_fidelity = eval_highest_fidelity

        self.blackbox_mfdgp_fitter_uncond = model.copy_uncond()

        # We check if we are already given the conditioned models

        if (model_cond is None):
            self.pareto_set, self.pareto_front, self.samples_objs, self.samples_cons = model.sample_and_store_pareto_solution()

            model.train_conditioned_mfdgps()

            self.blackbox_mfdgp_fitter_cond = model
        else:
            self.pareto_set = model_cond.pareto_set     # XXX next test, delete this line
            self.pareto_front = model_cond.pareto_front # XXX next test, delete this line
            # self.samples_objs = model_cond.samples_objs_last_fidelity # XXX next test, delete this line
            # self.samples_cons = model_cond.samples_cons_last_fidelity # XXX next test, delete this line

            self.blackbox_mfdgp_fitter_cond = model_cond

        self.num_fidelities = num_fidelities

        self.objectives = {}
        self.constraints = {}
        self.costs_blackboxes = {}
        for n_f in range(0, num_fidelities):
            self.objectives[ n_f ] = {}
            self.constraints[ n_f ] = {}
            
            self.costs_blackboxes[ n_f ] = {}            
            self.costs_blackboxes[ n_f ][ "total" ] = 0.0

    # def add_blackbox(self, fidelity: int, blackbox_name: str, is_constraint=False):
    def add_blackbox(self, fidelity: int, blackbox_name: str, cost_evaluation: float = 1.0, is_constraint=False):
            
        mfdgp_uncond = self.blackbox_mfdgp_fitter_uncond.get_model(blackbox_name, is_constraint=is_constraint )
        mfdgp_cond = self.blackbox_mfdgp_fitter_cond.get_model(blackbox_name, is_constraint=is_constraint )

        jes_mfdgp = _JES_MFDGP(fidelity, mfdgp_uncond, mfdgp_cond)

        if is_constraint:
            self.constraints[ fidelity ][ blackbox_name ] = jes_mfdgp
        else:
            self.objectives[ fidelity ][ blackbox_name ] = jes_mfdgp

        self.costs_blackboxes[ fidelity ][ "total" ] += cost_evaluation
        self.costs_blackboxes[ fidelity ][ blackbox_name ] = cost_evaluation

        return jes_mfdgp
    
    def decoupled_acq(self, X: Tensor, fidelity: int, blackbox_name: str, is_constraint=True) -> Tensor:

        if is_constraint:
            return self.constraints[ fidelity ][ blackbox_name ](X.double()) # / self.costs_blackboxes[ fidelity ][ blackbox_name ]
        else:
            return self.objectives[ fidelity ][ blackbox_name ](X.double()) # / self.costs_blackboxes[ fidelity ][ blackbox_name ]
    
    def coupled_acq(self, X: Tensor, fidelity: int) -> Tensor:

        acq = torch.zeros(size=(X.shape[ 0 ],))

        for name_obj, obj in self.objectives[ fidelity ].items():
            acq += obj(X.double()) # / self.costs_blackboxes[ fidelity ][ name_obj ]

        for name_con, con in self.constraints[ fidelity ].items():
            acq += con(X.double()) # / self.costs_blackboxes[ fidelity ][ name_con ]

        return acq

    def _get_nextpoint_coupled_highest_fidelity(self, iteration=None, verbose=False) -> Tensor:  # XXX delete choser part

        if verbose: assert (iteration is not None)

        fidelity_to_evaluate = self.num_fidelities - 1
        current_candidate, current_value = optimize_acqf(acq_function=lambda x: self.coupled_acq(x, fidelity=(self.num_fidelities - 1)), bounds=self.standard_bounds,
            q=1, num_restarts=5, raw_samples=200,  options={"maxiter": 200})
        current_value_weighted = current_value / self.costs_blackboxes[ 0 ][ "total" ]

        nextpoint = current_candidate[ 0, : ]
        if verbose: print("Iter:", iteration, "Acquisition: " + str(current_value_weighted.numpy()) + " Evaluating fidelity", fidelity_to_evaluate, "at", nextpoint.numpy())

        return nextpoint, fidelity_to_evaluate

    def _get_nextpoint_coupled(self, iteration=None, verbose=False) -> Tensor:

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

    def get_nextpoint_coupled(self, iteration=None, verbose=False) -> Tensor:

        if self.eval_highest_fidelity:
            return self._get_nextpoint_coupled_highest_fidelity(iteration=iteration, verbose=verbose)

        else:
            return self._get_nextpoint_coupled(iteration=iteration, verbose=verbose)



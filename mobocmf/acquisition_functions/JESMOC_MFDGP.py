
import copy
import numpy as np

import torch
import gpytorch

from torch import Tensor

# from botorch.acquisition.multi_objective import MultiObjectiveAnalyticAcquisitionFunction
# from botorch.acquisition.multi_objective.objective import AnalyticMultiOutputObjective
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

from mobocmf.util.blackbox_mfdgp_fitter import BlackBoxMFDGPFitter
from mobocmf.models.mfdgp import MFDGP


class _JES_MFDGP(AnalyticAcquisitionFunction):
    def __init__(
        self,
        mean,
        std,
        fidelity: int,
        mfdgp_uncond: MFDGP,
        mfdgp_cond: MFDGP,
        model: Model = None,
        maximize: bool = False,
    ) -> None:
        
        assert model is None
        
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        
        self.mean = mean
        self.std = std
        self.fidelity = fidelity

        self.mfdgp_uncond = mfdgp_uncond
        self.mfdgp_cond = mfdgp_cond
        

    # @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate JES_FMDGP
        """

        self.mfdgp_uncond.eval()
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_uncond, pred_variances_uncond = self.mfdgp_uncond.predict_for_acquisition(X, self.fidelity)
        self.mfdgp_uncond.train()

        # pred_mean_uncond = pred_means * std + self.mean
        pred_std_uncond  = np.sqrt(pred_variances_uncond) * self.std

        self.mfdgp_cond.eval()
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_cond, pred_variances_cond = self.mfdgp_cond.predict_for_acquisition(X, self.fidelity)
        self.mfdgp_cond.train()

        # pred_mean_cond = pred_means * self.std + self.mean
        pred_std_cond  = np.sqrt(pred_variances_cond) * self.std

        # pred_var_uncond = (pred_std_uncond[ 0 ])**2
        # pred_var_cond = (pred_std_cond[ 0 ])**2

        # return 0.5*(torch.log(pred_var_uncond) - torch.log(pred_var_cond))

        return torch.clamp(torch.log(pred_std_uncond[ 0 ]) - torch.log(pred_std_cond[ 0 ]), min=0.0)
        

class JESMOC_MFDGP():
    def __init__(
        self,
        model: BlackBoxMFDGPFitter,
        num_fidelities: int = 1,
        maximize: bool = False,
    ) -> None:
        assert maximize == False

        self.maximize = maximize

        self.blackbox_mfdgp_fitter_uncond = model.copy_uncond()

        self.pareto_set, self.pareto_front, self.samples_objs = model.sample_and_store_pareto_solution()

        model.train_conditioned_mfdgps()

        self.blackbox_mfdgp_fitter_cond = model

        self.num_fidelities = num_fidelities

        self.objectives = {}
        self.constraints = {}
        for n_f in range(0, num_fidelities):
            self.objectives[ n_f ] = {}
            self.constraints[ n_f ] = {}

    def add_blackbox(self, mean, std, fidelity: int, blackbox_name: str, is_constraint=False):
            
        mfdgp_uncond = self.blackbox_mfdgp_fitter_uncond.get_model( blackbox_name, is_constraint=is_constraint )
        mfdgp_cond = self.blackbox_mfdgp_fitter_cond.get_model( blackbox_name, is_constraint=is_constraint )

        jes_mfdgp = _JES_MFDGP(mean, std, fidelity, mfdgp_uncond, mfdgp_cond)

        if is_constraint:
            self.constraints[ fidelity ][ blackbox_name ] = jes_mfdgp
        else:
            self.objectives[ fidelity ][ blackbox_name ] = jes_mfdgp

        return jes_mfdgp
    
    def decoupled_acq(self, X: Tensor, fidelity: int, blackbox_name: str, is_constraint=True) -> Tensor:

        if is_constraint:
            return self.constraints[ fidelity ][ blackbox_name ](X)
        else:
            return self.objectives[ fidelity ][ blackbox_name ](X)
    
    def coupled_acq(self, X: Tensor, fidelity: int) -> Tensor:

        acq = torch.zeros(size=(X.shape[ 0 ],))

        for obj in self.objectives[ fidelity ].values():
            acq += obj(X)

        for con in self.constraints[ fidelity ].values():
            acq += con(X)

        return acq




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
        
        # self.mean = mean
        # self.std = std
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

        self.mfdgp_cond.eval()
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_cond, pred_variances_cond = self.mfdgp_cond.predict_for_acquisition(X, self.fidelity)
        self.mfdgp_cond.train()

#        return 0.5 * torch.clamp(torch.log(pred_variances_uncond) - torch.log(pred_variances_cond), min=0.0) 
        return torch.clamp(pred_variances_uncond - pred_variances_cond, min=0.0)
        

class JESMOC_MFDGP():
    def __init__(
        self,
        model: BlackBoxMFDGPFitter,
        num_fidelities: int = 1,
        model_cond: BlackBoxMFDGPFitter = None,
    ) -> None:

        self.blackbox_mfdgp_fitter_uncond = model.copy_uncond()

        # We check if we are already given the conditioned models

        if (model_cond is None):
            self.pareto_set, self.pareto_front, self.samples_objs = model.sample_and_store_pareto_solution()

            model.train_conditioned_mfdgps()

            self.blackbox_mfdgp_fitter_cond = model
        else:
            self.pareto_set = model_cond.pareto_set
            self.pareto_front = model_cond.pareto_front
            self.samples_objs = model_cond.samples_objs

            self.blackbox_mfdgp_fitter_cond = model_cond

        self.num_fidelities = num_fidelities

        self.objectives = {}
        self.constraints = {}
        for n_f in range(0, num_fidelities):
            self.objectives[ n_f ] = {}
            self.constraints[ n_f ] = {}

    def add_blackbox(self, fidelity: int, blackbox_name: str, is_constraint=False):
            
        mfdgp_uncond = self.blackbox_mfdgp_fitter_uncond.get_model(blackbox_name, is_constraint=is_constraint )
        mfdgp_cond = self.blackbox_mfdgp_fitter_cond.get_model(blackbox_name, is_constraint=is_constraint )

        jes_mfdgp = _JES_MFDGP(fidelity, mfdgp_uncond, mfdgp_cond)

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



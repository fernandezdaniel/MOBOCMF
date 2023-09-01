
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

class RANDOM_choice():
    def __init__(
        self,
        num_fidelities: int = 1,
        l_costs_fidelities: list = [1],
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



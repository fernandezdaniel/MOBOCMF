import torch

from torch import Tensor

class Random_choice():
    def __init__(
        self,
        input_size = None,
        num_fidelities: int = 1,
        seed = None,
    ) -> None:
        
        self.input_size = input_size

        self.num_fidelities = num_fidelities

        self.seed = seed

        torch.manual_seed(self.seed)
        
        self.costs_blackboxes = {}
        for n_f in range(0, num_fidelities):            
            self.costs_blackboxes[ n_f ] = {}
            self.costs_blackboxes[ n_f ][ "total" ] = 0.0

        self.coupled_costs_fidelities = torch.zeros(self.num_fidelities)
        self.total_cost_fidelities = 0.0

    def add_blackbox(self, fidelity: int, blackbox_name: str, cost_evaluation: float = 1.0):

        self.costs_blackboxes[ fidelity ][ blackbox_name ] = cost_evaluation

        self.coupled_costs_fidelities[ fidelity ] += cost_evaluation
        self.total_cost_fidelities += cost_evaluation
    
    def decoupled_acq(self, X: Tensor, fidelity: int, blackbox_name) -> Tensor:

        return torch.rand(size=(X.shape[ 0 ], ))
    
    def coupled_acq(self, X: Tensor, fidelity: int) -> Tensor:

        return torch.rand(size=(X.shape[ 0 ], ))

    def get_nextpoint_coupled(self, iteration=None, verbose=False) -> Tensor:

        fidelities = torch.arange(self.num_fidelities)
        probs_fidelities = self.coupled_costs_fidelities / self.total_cost_fidelities

        nextpoint = torch.rand(size=(self.input_size, ))
        fidelity_to_evaluate = fidelities[ torch.multinomial(probs_fidelities, 1).item() ].item()

        if verbose: print("Iter:", iteration, " Evaluating fidelity", fidelity_to_evaluate, "at", nextpoint.numpy())

        return nextpoint, fidelity_to_evaluate


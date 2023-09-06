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
    
    def decoupled_acq(self, X: Tensor) -> Tensor:

        return torch.rand(size=X.shape)
    
    def coupled_acq(self, X: Tensor) -> Tensor:

        return torch.rand(size=X.shape)

    def get_nextpoint_coupled(self) -> Tensor:

        fidelities = torch.arange(self.num_fidelities)
        probs_fidelities = self.coupled_costs_fidelities / self.total_cost_fidelities

        return torch.rand(size=(1, self.input_size)), fidelities[ torch.multinomial(probs_fidelities, 1).item() ]



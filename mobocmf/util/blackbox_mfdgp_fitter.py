
import warnings
import numpy as np
import torch
import gpytorch

from scipy.stats import norm
from torch.utils.data import TensorDataset, DataLoader

from mobocmf.models.mfdgp import MFDGP
from mobocmf.mlls.variational_elbo_mf import VariationalELBOMF
from mobocmf.util.moop import MOOP, NotFeasiblePoints

from copy import deepcopy

dist = torch.distributions.normal.Normal(0, 1) 


class MFDGPHandler():

    MAX_TRIES_FOR_FEASIBLE_GRID = 10

    def __init__(self, x_train, y_train, fidelities_train, num_fidelities, batch_size):
        
        self.mfdgp = MFDGP(x_train, y_train, fidelities_train, num_fidelities=num_fidelities)
        self.mfdgp.double() # We use double precission to avoid numerical problems
        self.elbo = VariationalELBOMF(self.mfdgp, x_train.shape[-2], num_fidelities=num_fidelities)

        self.train_dataset = TensorDataset(x_train, y_train, fidelities_train)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.iter_train_loader = None # iter(self.train_loader) # DFS: If we comment this initialization we can pickle the BlackBoxMFDGPFitter objects
        self.num_data = x_train.shape[ 0 ]
        self.num_fidelities = num_fidelities

class BlackBoxMFDGPFitter():

    def __init__(self, num_fidelities, batch_size, lr_1=0.003, lr_2=0.001, num_epochs_1=5000, num_epochs_2=15000,
                 pareto_set_size=50, opt_grid_size=1000, eps=1e-8, decoupled_evals=False):
        
        # assert decoupled_evals is False, "This object is not currently prepared for a decoupled evaluation setting."

        self.num_obj = 0
        self.num_con = 0

        self.models_uncond_trained = False

        self.mfdgp_handlers_objs = {}
        self.mfdgp_handlers_cons = {}

        self.thresholds_cons = torch.tensor([], dtype=torch.double) # thresholds constraints

        self.x_train = None
        self.objs_train = torch.tensor([], dtype=torch.double)
        self.cons_train = torch.tensor([], dtype=torch.double)

        self.num_fidelities = num_fidelities

        self.batch_size = batch_size
        self.points_to_sample = batch_size

        self.lr_1 = lr_1
        self.lr_2 = lr_2

        self.num_epochs_1 = num_epochs_1
        self.num_epochs_2 = num_epochs_2

        self.pareto_set_size = pareto_set_size

        self.opt_grid_size = opt_grid_size

        self.eps = eps

        self.decoupled_evals = decoupled_evals

    def initialize_mfdgp(self, x_train, y_train, fidelities, blackbox_name, threshold_constraint=0.0, is_constraint=False):

        if self.x_train is None:
            self.x_train = x_train
        else:
            assert torch.equal(self.x_train, x_train), "The inputs for this new mfdgp do not match with inputs for previous mfdgp models. This class is not currently prepared for a decoupled evaluation setting."
        
        if is_constraint:

            self.cons_train = torch.cat((self.cons_train, y_train), 1)

            self.mfdgp_handlers_cons[ blackbox_name ] = MFDGPHandler(x_train, y_train, fidelities, self.num_fidelities, self.batch_size)
            
            self.thresholds_cons = torch.cat((self.thresholds_cons, torch.tensor([threshold_constraint])), 0)

            self.num_con += 1

        else:

            self.objs_train = torch.cat((self.objs_train, y_train), 1)

            self.mfdgp_handlers_objs[ blackbox_name ] = MFDGPHandler(x_train, y_train, fidelities, self.num_fidelities, self.batch_size)

            self.num_obj += 1
    
    def _train_mfdgp(self, func_update_model, fix_variational_hypers, num_epochs, lr):

        l_opt_objs = []
        l_opt_cons = []

        for handler_obj in self.mfdgp_handlers_objs.values():

            # handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_obj.mfdgp.fix_params(fix_variational_hypers)

            l_opt_objs.append(torch.optim.Adam([ {'params': handler_obj.mfdgp.parameters()} ], lr=lr))

        for handler_con in self.mfdgp_handlers_cons.values():

            # handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_con.mfdgp.fix_params(fix_variational_hypers)

            l_opt_cons.append(torch.optim.Adam([ {'params': handler_con.mfdgp.parameters()} ], lr=lr)) # Pasara qui los aprametros de todos los mdoelos (para el loss general)

        for (handler_obj, optimizer, n_obj) in zip(self.mfdgp_handlers_objs.values(), l_opt_objs, np.arange(len(self.mfdgp_handlers_objs))):
            
            for i in range(num_epochs):
                
                loss_iter = func_update_model(handler_obj.mfdgp, handler_obj.elbo, optimizer, handler_obj.train_loader)

                print("[OBJ: ", n_obj, "] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item())
            
        for (handler_con, optimizer, n_con) in zip(self.mfdgp_handlers_cons.values(), l_opt_cons, np.arange(len(self.mfdgp_handlers_cons))):

            for i in range(num_epochs):
                
                loss_iter = func_update_model(handler_con.mfdgp, handler_con.elbo, optimizer, handler_con.train_loader)

                print("[CON: ", n_con, "] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item())

    def train_mfdgps(self):

        def _update_model(model, elbo, optimizer, train_loader):

            loss_iter = 0.0

            for (x_batch, y_batch, fidelities) in train_loader:

                with gpytorch.settings.num_likelihood_samples(1):
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -elbo(output, y_batch.T, fidelities)
                    loss.backward()
                    optimizer.step()
                    loss_iter += loss

            return loss_iter
        
        self._train_mfdgp(_update_model, fix_variational_hypers=True, num_epochs=self.num_epochs_1, lr=self.lr_1)
        self._train_mfdgp(_update_model, fix_variational_hypers=False, num_epochs=self.num_epochs_2, lr=self.lr_2)

        self.models_uncond_trained = True
        
    def sample_and_store_pareto_solution(self):

        l_samples_objs = []

        for handler_obj in self.mfdgp_handlers_objs.values():
            l_samples_objs.append(handler_obj.mfdgp.sample_function_from_each_layer()[ -1 ])


        for _ in range(MFDGPHandler.MAX_TRIES_FOR_FEASIBLE_GRID): # DFS: Is this right? Sometimes there are no feasible point so
            l_samples_cons = []
            
            for handler_con in self.mfdgp_handlers_cons.values():
                l_samples_cons.append(handler_con.mfdgp.sample_function_from_each_layer()[ -1 ])

            inputs = self.x_train

            global_optimizer = MOOP(l_samples_objs,
                                    l_samples_cons,
                                    input_dim=inputs.shape[ 1 ],
                                    grid_size=self.opt_grid_size,
                                    pareto_set_size=self.pareto_set_size)

            # self.pareto_set, self.pareto_front = global_optimizer.compute_pareto_solution_from_samples(inputs)

            # return self.pareto_set, self.pareto_front

            if (res := global_optimizer.compute_pareto_solution_from_samples(inputs)) is not None:
                self.pareto_set, self.pareto_front, self.samples_objs = res
                return self.pareto_set, self.pareto_front, self.samples_objs

        raise NotFeasiblePoints("[ERROR] No feasible points were found in the constraint space! # tries: %d." % MFDGPHandler.MAX_TRIES_FOR_FEASIBLE_GRID)

    def loss_theta_factors(self, cs_mean, cs_var):

        gamma_c_star = (cs_mean - self.thresholds_cons) / torch.sqrt(cs_var)

        cdf_gamma_c_star = dist.cdf(gamma_c_star)

        return torch.sum(np.log(1.0 - self.eps) * cdf_gamma_c_star + np.log(self.eps) * (1.0 - cdf_gamma_c_star))

    def loss_omega_factors(self, fs_mean, fs_var, cs_mean, cs_var, pareto_front):

        gamma_c      = (cs_mean - self.thresholds_cons) / torch.sqrt(cs_var)
        gamma_f_star = (pareto_front[ : , : , None] - fs_mean) / torch.sqrt(fs_var)

        cdf_gamma_c_times_cdf_gamma_f_star = torch.prod(dist.cdf(gamma_c), 1) * torch.prod(dist.cdf(gamma_f_star), 2) # DHL to check!!!

        return torch.sum(np.log(1.0 - self.eps) * cdf_gamma_c_times_cdf_gamma_f_star
                         + np.log(self.eps) * (1.0 - cdf_gamma_c_times_cdf_gamma_f_star))

    def _train_conditioned_mfdgps(self, func_update_model, fix_variational_hypers, num_epochs, lr):

        params = list()

        for handler_obj in self.mfdgp_handlers_objs.values():
            
            # handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_obj.mfdgp.fix_params(fix_variational_hypers)
            params = params + list(handler_obj.mfdgp.parameters())

        for handler_con in self.mfdgp_handlers_cons.values():
            
            # handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_con.mfdgp.fix_params(fix_variational_hypers)
            params = params + list(handler_con.mfdgp.parameters())

        optimizer = torch.optim.Adam([ {'params': params } ], lr=lr)

        for i in range(num_epochs):

            loss_iter = func_update_model(self.mfdgp_handlers_objs.values(), self.mfdgp_handlers_cons.values(), optimizer)

            print("Iter:", i, "/", num_epochs, ". Neg. ELBO per iter:", loss_iter.item())

    def train_conditioned_mfdgps(self):

        def _update_conditioned_models(handlers_objs, handlers_cons, optimizer):

            loss = 0.0
            optimizer.zero_grad()
           
            x_tilde = torch.rand(size=(10, self.pareto_set.shape[ 1 ])).double() # sampling: x_batch_pints * x_dim
                
            for i, handler_objs in enumerate(handlers_objs):

                try:
                    (x_batch, y_batch, fidelities) = next(handler_objs.iter_train_loader)
                except:
                    handler_objs.iter_train_loader = iter(handler_objs.train_loader)
                    (x_batch, y_batch, fidelities) = next(handler_objs.iter_train_loader)

                with gpytorch.settings.num_likelihood_samples(1):
                    output = handler_objs.mfdgp(x_batch)
                    loss += -handler_objs.elbo(output, y_batch.T, fidelities) / x_batch.shape[ 0 ] * handler_objs.num_data

                    output = handler_objs.mfdgp(self.pareto_set)
                    pareto_fidelities = torch.ones(size=(self.pareto_front.shape[ 0 ], 1)) * (handler_objs.num_fidelities - 1)
                    loss += -handler_objs.elbo(output, self.pareto_front[ :, i : (i + 1) ].T, pareto_fidelities, include_kl_term = False)

            for i, handler_cons in enumerate(handlers_cons):

                try:
                    (x_batch, y_batch, fidelities) = next(handler_cons.iter_train_loader)
                except:
                    handler_cons.iter_train_loader = iter(handler_cons.train_loader)
                    (x_batch, y_batch, fidelities) = next(handler_cons.iter_train_loader)

                with gpytorch.settings.num_likelihood_samples(1):
                    output = handler_cons.mfdgp(x_batch)
                    loss += -handler_cons.elbo(output, y_batch.T, fidelities) / x_batch.shape[ 0 ] * handler_cons.num_data

                    # We add factors that ensure positive constraints at the pareto points. These are the theta factors.

                    output = handler_cons.mfdgp(self.pareto_set)[ handler_cons.num_fidelities - 1 ]
                    loss += -self.loss_theta_factors(output.mean, output.variance) 

            # We add the omega factors

            means_c_x_tilde = torch.tensor([], dtype=torch.double)
            means_f_x_tilde = torch.tensor([], dtype=torch.double)
            vars_c_pareto = torch.tensor([], dtype=torch.double)
            vars_c_x_tilde = torch.tensor([], dtype=torch.double)
            vars_f_x_tilde = torch.tensor([], dtype=torch.double)

            for i, handler_objs in enumerate(handlers_objs):

                with gpytorch.settings.num_likelihood_samples(1):
                    output = handler_objs.mfdgp(x_tilde)[ handler_objs.num_fidelities - 1 ]

                    mean_f_x_tilde, var_f_x_tilde = output.mean, output.variance
                    means_f_x_tilde = torch.cat((means_f_x_tilde, mean_f_x_tilde[None, :]), 0)
                    vars_f_x_tilde = torch.cat((vars_f_x_tilde, var_f_x_tilde[None, :]), 0)

            for i, handler_cons in enumerate(handlers_cons):

                with gpytorch.settings.num_likelihood_samples(1):
                    output = handler_cons.mfdgp(x_tilde)[ handler_cons.num_fidelities - 1 ] # We pass the data through the DGP to obtain the predictions

                    mean_c_x_tilde, var_c_x_tilde = output.mean, output.variance
                    means_c_x_tilde = torch.cat((means_c_x_tilde, mean_c_x_tilde[None, :]), 0)
                    vars_c_x_tilde = torch.cat((vars_c_x_tilde, var_c_x_tilde[None, :]), 0)

            loss += -self.loss_omega_factors(means_f_x_tilde, vars_f_x_tilde, means_c_x_tilde, vars_c_x_tilde, self.pareto_front)  

            loss.backward()
            optimizer.step()

            return loss

        self._train_conditioned_mfdgps(_update_conditioned_models, fix_variational_hypers=True, num_epochs=self.num_epochs_1, lr=self.lr_1)
        self._train_conditioned_mfdgps(_update_conditioned_models, fix_variational_hypers=False, num_epochs=self.num_epochs_2, lr=self.lr_2)

        for handler_obj in self.mfdgp_handlers_objs.values():
            handler_obj.iter_train_loader = None

        for handler_con in self.mfdgp_handlers_cons.values():
            handler_con.iter_train_loader = None

    def mfdgps_to_train_mode(self):
        for handler in self.mfdgp_handlers_objs.values():
            handler.mfdgp.train()

        for handler in self.mfdgp_handlers_cons.values():
            handler.mfdgp.train()

    def mfdgps_to_eval_mode(self):
        for handler in self.mfdgp_handlers_objs.values():
            handler.mfdgp.eval()

        for handler in self.mfdgp_handlers_cons.values():
            handler.mfdgp.train()

    def copy_uncond(self):

        if self.models_uncond_trained is False: 
            warnings.warn("(Warning) The mfdgp models have not been trained yet.")

        self.mfdgps_to_train_mode()

        self_copy = deepcopy(self)

        return self_copy


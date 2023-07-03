
import warnings
import numpy as np
import torch
import gpytorch

from scipy.stats import norm
from torch.utils.data import TensorDataset, DataLoader

from mobocmf.models.mfdgp import MFDGP, TL
from mobocmf.mlls.variational_elbo_mf import VariationalELBOMF
from mobocmf.util.moop import MOOP, NotFeasiblePoints

from copy import deepcopy

dist = torch.distributions.normal.Normal(0, 1) 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MFDGPHandlerPlot():

    MAX_TRIES_FOR_FEASIBLE_GRID = 10

    def __init__(self, x_train, y_train, fidelities_train, num_fidelities, batch_size, type_lengthscale):
        
        self.mfdgp = MFDGP(x_train, y_train, fidelities_train, num_fidelities=num_fidelities, type_lengthscale=type_lengthscale)
        self.mfdgp.double() # We use double precission to avoid numerical problems
        self.elbo = VariationalELBOMF(self.mfdgp, x_train.shape[-2], num_fidelities=num_fidelities)

        self.train_dataset = TensorDataset(x_train, y_train, fidelities_train)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.iter_train_loader = None # iter(self.train_loader) # DFS: If we comment this initialization we can pickle the BlackBoxMFDGPFitter objects
        self.num_data = x_train.shape[ 0 ]
        self.num_fidelities = num_fidelities

class BlackBoxMFDGPFitterPlot():

    def __init__(self, num_fidelities, batch_size, lr_1=0.003, lr_2=0.001, num_epochs_1=5000, num_epochs_2=15000,
                 pareto_set_size=50, opt_grid_size=1000, eps=1e-8, decoupled_evals=False, type_lengthscale=TL.MEDIAN,
                 step_plot=50):
        
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
        self.pareto_set = None
        self.pareto_front = None

        self.opt_grid_size = opt_grid_size

        self.eps = eps

        self.decoupled_evals = decoupled_evals

        self.type_lengthscale = type_lengthscale

        self.step_plot = step_plot

        self.axs = None
        self.lower_limit = None
        self.upper_limit = None
        self.func_con1_mf0 = None
        self.func_con1_mf1 = None
        self.x_mf0 = None
        self.x_mf1 = None
        self.con1_mean_mf0 = None
        self.con1_mean_mf1 = None
        self.con1_std_mf0 = None
        self.con1_std_mf1 = None
        self.con1_train_mf0 = None
        self.con1_train_mf1 = None


    def initialize_mfdgp(self, x_train, y_train, fidelities, blackbox_name, threshold_constraint=0.0, is_constraint=False):

        if self.x_train is None:
            self.x_train = x_train
        else:
            assert torch.equal(self.x_train, x_train), "The inputs for this new mfdgp do not match with inputs for previous mfdgp models. This class is not currently prepared for a decoupled evaluation setting."
        
        if is_constraint:

            self.cons_train = torch.cat((self.cons_train, y_train), 1)

            self.mfdgp_handlers_cons[ blackbox_name ] = MFDGPHandlerPlot(x_train, y_train, fidelities, self.num_fidelities, self.batch_size, type_lengthscale=self.type_lengthscale)
            
            self.thresholds_cons = torch.cat((self.thresholds_cons, torch.tensor([threshold_constraint])), 0)

            self.num_con += 1

        else:

            self.objs_train = torch.cat((self.objs_train, y_train), 1)

            self.mfdgp_handlers_objs[ blackbox_name ] = MFDGPHandlerPlot(x_train, y_train, fidelities, self.num_fidelities, self.batch_size, type_lengthscale=self.type_lengthscale)

            self.num_obj += 1

    def _set_parameters_to_plot_con1(self,
                                     lower_limit, upper_limit,
                                     func_con1_mf0, func_con1_mf1,
                                     x_mf0, x_mf1,
                                     con1_mean_mf0, con1_mean_mf1,
                                     con1_std_mf0, con1_std_mf1,
                                     con1_train_mf0, con1_train_mf1):

        # Crear los espacios para los subplots
        gs = gridspec.GridSpec(3, 4, width_ratios=[4, 1, 1, 1], height_ratios=[1, 1, 1])

        # Inicializar la figura y los subplots
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax0 = plt.subplot(gs[:, 0]) # MFDGP
        ax1 = plt.subplot(gs[0, 1]) # 
        ax2 = plt.subplot(gs[1, 1]) # 
        ax3 = plt.subplot(gs[2, 1]) # 
        ax4 = plt.subplot(gs[0, 2]) # 
        ax5 = plt.subplot(gs[1, 2]) # 
        ax6 = plt.subplot(gs[2, 2]) # 
        ax7 = plt.subplot(gs[0, 3]) # 
        ax8 = plt.subplot(gs[1, 3]) # 
        ax9 = plt.subplot(gs[2, 3]) # 

        self.axs=[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.func_con1_mf0 = func_con1_mf0
        self.func_con1_mf1 = func_con1_mf1
        self.x_mf0 = x_mf0
        self.x_mf1 = x_mf1
        self.con1_mean_mf0 = con1_mean_mf0
        self.con1_mean_mf1 = con1_mean_mf1
        self.con1_std_mf0 = con1_std_mf0
        self.con1_std_mf1 = con1_std_mf1
        self.con1_train_mf0 = con1_train_mf0
        self.con1_train_mf1 = con1_train_mf1


    def _prepare_training_mode_with_plots(self):
        self.loss_c_iters, self.nelbo_c_iters, self.kl_c_iters, self.loss_iters = [], [], [], []

        self.iters_uncond = []
        self.iters_cond = []
        self.vals_param_k_x_1_outputscale, self.vals_param_k_x_1_lengthscale = [], []
        self.vals_param_k_lin_variance = []
        self.vals_param_k_f_outputscale, self.vals_param_k_f_lengthscale = [], []
        self.vals_param_k_x_2_outputscale, self.vals_param_k_x_2_lengthscale = [], []
        self.vals_param_hidden_layer_likelihood_1_noise = []

    def _plot_mfdgp_and_params(self, loss_c_iter, kl_c_iter, handler_con, loss_iter=None, pareto_set=None, pareto_front_vals=None):

        if loss_iter is None: 
            if len(self.iters_uncond) == 0:
                self.iters_uncond.append(0)
            else:
                self.iters_uncond.append(self.iters_uncond[-1] + self.step_plot)
        else:
            if len(self.iters_cond) == 0:
                self.iters_cond.append(0)
            else:
                self.iters_cond.append(self.iters_cond[-1] + self.step_plot)

        self.loss_c_iters.append(loss_c_iter.item())
        self.nelbo_c_iters.append(loss_c_iter.item() - kl_c_iter.item())
        self.kl_c_iters.append(kl_c_iter.item())
        if loss_iter is not None: self.loss_iters.append(loss_iter.item())

        x_func = np.linspace(self.lower_limit, self.upper_limit, 1000)[:, None]
        spacing = torch.linspace(self.lower_limit, self.upper_limit, 200).double()[ : , None ]

        handler_con.mfdgp.eval()
        con1_pred_mean_mf0, con1_pred_std_mf0 = compute_moments_mfdgp(handler_con.mfdgp, spacing, mean=self.con1_mean_mf0, std=self.con1_std_mf0, fidelity=0)
        con1_pred_mean_mf1, con1_pred_std_mf1 = compute_moments_mfdgp(handler_con.mfdgp, spacing, mean=self.con1_mean_mf1, std=self.con1_std_mf1, fidelity=1)
        handler_con.mfdgp.train()

        # Actualizar el plot principal
        self.axs[ 0 ].cla()
        if loss_iter is not None: 
            plot_model(self.axs[ 0 ], spacing.numpy(),
                    x_func, x_func,
                    self.func_con1_mf0(x_func), self.func_con1_mf1(x_func),
                    self.x_mf0, self.x_mf1,
                    self.con1_mean_mf0, self.con1_mean_mf1,
                    self.con1_std_mf0, self.con1_std_mf1,
                    self.con1_train_mf0, self.con1_train_mf1,
                    con1_pred_mean_mf0, con1_pred_mean_mf1,
                    con1_pred_std_mf0, con1_pred_std_mf1,
                    pareto_set=self.pareto_set, pareto_front_vals=self.pareto_front[ : , 0 ]*0.0, cons=True)
        else:
            plot_model(self.axs[ 0 ], spacing.numpy(),
                    x_func, x_func,
                    self.func_con1_mf0(x_func), self.func_con1_mf1(x_func),
                    self.x_mf0, self.x_mf1,
                    self.con1_mean_mf0, self.con1_mean_mf1,
                    self.con1_std_mf0, self.con1_std_mf1,
                    self.con1_train_mf0, self.con1_train_mf1,
                    con1_pred_mean_mf0, con1_pred_mean_mf1,
                    con1_pred_std_mf0, con1_pred_std_mf1,
                    cons=True)

        
        # Actualizar los subplots de los parámetros
        self.axs[ 1 ].cla()
        self.axs[ 1 ].plot(self.iters_uncond, self.loss_c_iters, 'b-', label="Loss c1")
        self.axs[ 1 ].plot(self.iters_uncond, self.nelbo_c_iters, 'g-', label="NELBO c1")
        self.axs[ 1 ].plot(self.iters_uncond, self.kl_c_iters, 'r-', label="KL c1")
        if loss_iter is not None: self.axs[ 1 ].plot(self.iters_uncond, self.loss_c_iters, 'r-', label="Loss")
        self.axs[ 1 ].legend()

        param_k_x_1_outputscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[0].kernels[0].outputscale.item()
        param_k_x_1_lengthscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[0].kernels[0].base_kernel.lengthscale.item()
        param_k_lin_variance = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[0].kernels[1].kernels[0].variance.item()
        param_k_f_outputscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[0].kernels[1].kernels[1].outputscale.item()
        param_k_f_lengthscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[0].kernels[1].kernels[1].base_kernel.lengthscale.item()
        param_k_x_2_outputscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[1].outputscale.item()
        param_k_x_2_lengthscale = handler_con.mfdgp.hidden_layer_1.covar_module.kernels[1].base_kernel.lengthscale.item()
        param_hidden_layer_likelihood_1_noise = handler_con.mfdgp.hidden_layer_likelihood_1.noise_covar.noise.item()

        self.vals_param_k_x_1_outputscale.append(param_k_x_1_outputscale)
        self.vals_param_k_x_1_lengthscale.append(param_k_x_1_lengthscale)
        self.vals_param_k_lin_variance.append(param_k_lin_variance)
        self.vals_param_k_f_outputscale.append(param_k_f_outputscale)
        self.vals_param_k_f_lengthscale.append(param_k_f_lengthscale)
        self.vals_param_k_x_2_outputscale.append(param_k_x_2_outputscale)
        self.vals_param_k_x_2_lengthscale.append(param_k_x_2_lengthscale)
        self.vals_param_hidden_layer_likelihood_1_noise.append(param_hidden_layer_likelihood_1_noise)

        self.axs[ 2 ].cla()
        self.axs[ 2 ].plot(self.iters_uncond, self.vals_param_k_x_1_outputscale, 'b-')
        self.axs[ 2 ].set_title('k_x_1 outputscale')

        self.axs[ 3 ].cla()
        self.axs[ 3 ].plot(self.iters_uncond, self.vals_param_k_x_1_lengthscale, 'b-')
        self.axs[ 3 ].set_title('k_x_1 lengthscale')
        
        self.axs[ 4 ].cla()
        self.axs[ 4 ].plot(self.iters_uncond, self.vals_param_k_lin_variance, 'b-')
        self.axs[ 4 ].set_title('k_lin variance')
        
        self.axs[ 5 ].cla()
        self.axs[ 5 ].plot(self.iters_uncond, self.vals_param_k_f_outputscale, 'b-')
        self.axs[ 5 ].set_title('k_f outputscale')
        
        self.axs[ 6 ].cla()
        self.axs[ 6 ].plot(self.iters_uncond, self.vals_param_k_f_lengthscale, 'b-')
        self.axs[ 6 ].set_title('k_f lengthscale')
        
        self.axs[ 7 ].cla()
        self.axs[ 7 ].plot(self.iters_uncond, self.vals_param_k_x_2_outputscale, 'b-')
        self.axs[ 7 ].set_title('k_x_2 outputscale')
        
        self.axs[ 8 ].cla()
        self.axs[ 8 ].plot(self.iters_uncond, self.vals_param_k_x_2_lengthscale, 'b-')
        self.axs[ 8 ].set_title('k_x_2 lengthscale')
        
        self.axs[ 9 ].cla()
        self.axs[ 9 ].plot(self.iters_uncond, self.vals_param_hidden_layer_likelihood_1_noise, 'b-')
        self.axs[ 9 ].set_title('hidden_layer_likelihood_1 noise')

        if loss_iter is None: plt.suptitle("Iters uncond: " + str(self.iters_uncond[-1]))
        else:  plt.suptitle("Iters uncond: " + str(self.iters_uncond[-1]) + " ; Iters cond:" + str(self.iters_cond[-1]) )
        
        self.fig.canvas.draw()
        
        plt.pause(0.3)

        if len(self.iters_uncond) == 1: input()

    def _train_mfdgp_and_plot(self, func_update_model, fix_variational_hypers, num_epochs, lr, name_blackbox_to_plot):

        l_opt_objs = []
        l_opt_cons = []

        for handler_obj in self.mfdgp_handlers_objs.values():

            handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)

            l_opt_objs.append(torch.optim.Adam([ {'params': handler_obj.mfdgp.parameters()} ], lr=lr))

        for handler_con in self.mfdgp_handlers_cons.values():

            handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)

            l_opt_cons.append(torch.optim.Adam([ {'params': handler_con.mfdgp.parameters()} ], lr=lr)) # Pasara qui los aprametros de todos los mdoelos (para el loss general)

        for (handler_obj, optimizer, n_obj) in zip(self.mfdgp_handlers_objs.values(), l_opt_objs, np.arange(len(self.mfdgp_handlers_objs))):
            
            for i in range(num_epochs):
                
                loss_iter, kl_iter = func_update_model(handler_obj.mfdgp, handler_obj.elbo, optimizer, handler_obj.train_loader)

                print("[OBJ: ", n_obj, "] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item(), "\t KL per epoch:", kl_iter.item())
            
        for (item_dict_handlers_con, optimizer, n_con) in zip(self.mfdgp_handlers_cons.items(), l_opt_cons, np.arange(len(self.mfdgp_handlers_cons))):
            
            handler_name, handler_con = item_dict_handlers_con

            for i in range(num_epochs+1):
                
                loss_iter, kl_iter = func_update_model(handler_con.mfdgp, handler_con.elbo, optimizer, handler_con.train_loader)

                print("[CON: ", n_con, "] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item(), "\t KL per epoch:", kl_iter.item())

                if name_blackbox_to_plot == handler_name:

                    if i % self.step_plot == 0:

                        self._plot_mfdgp_and_params(loss_iter, kl_iter, handler_con)

    def train_mfdgps_and_plot(self, name_blackbox_to_plot, plt_show=False):

        def _update_model(model, elbo, optimizer, train_loader):

            loss_iter = 0.0
            kl_iter = 0.0

            for (x_batch, y_batch, fidelities) in train_loader:

                with gpytorch.settings.num_likelihood_samples(1):
                    optimizer.zero_grad()
                    output = model(x_batch)
                    res  = elbo(output, y_batch.T, fidelities)
                    loss, kl = -res[0], res[1]
                    loss.backward()
                    optimizer.step()
                    loss_iter += loss
                    kl_iter += kl

            return loss_iter, kl_iter

        self._prepare_training_mode_with_plots()

        self._train_mfdgp_and_plot(_update_model,
                                   fix_variational_hypers=True,
                                   num_epochs=self.num_epochs_1,
                                   lr=self.lr_1,
                                   name_blackbox_to_plot=name_blackbox_to_plot)
        self._train_mfdgp_and_plot(_update_model,
                                   fix_variational_hypers=False,
                                   num_epochs=self.num_epochs_2,
                                   lr=self.lr_2,
                                   name_blackbox_to_plot=name_blackbox_to_plot)

        self.models_uncond_trained = True

        if plt_show: plt.show()
        
    def sample_and_store_pareto_solution(self):

        l_samples_objs = []

        for handler_obj in self.mfdgp_handlers_objs.values():
            l_samples_objs.append(handler_obj.mfdgp.sample_function_from_each_layer()[ -1 ])


        for _ in range(MFDGPHandlerPlot.MAX_TRIES_FOR_FEASIBLE_GRID): # DFS: Is this right? Sometimes there are no feasible point so
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

        raise NotFeasiblePoints("[ERROR] No feasible points were found in the constraint space! # tries: %d." % MFDGPHandlerPlot.MAX_TRIES_FOR_FEASIBLE_GRID)

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

    def _train_conditioned_mfdgps_and_plot(self, func_update_model, fix_variational_hypers, num_iters, lr,
                                           name_blackbox_to_plot):

        params = list()

        for handler_obj in self.mfdgp_handlers_objs.values():
            
            # handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_obj.mfdgp.fix_variational_hypers_cond(fix_variational_hypers)
            params = params + list(handler_obj.mfdgp.parameters())

        for handler_con in self.mfdgp_handlers_cons.values():
            
            # handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)
            handler_con.mfdgp.fix_variational_hypers_cond(fix_variational_hypers)
            params = params + list(handler_con.mfdgp.parameters())

        optimizer = torch.optim.Adam([ {'params': params } ], lr=lr)

        for i in range(num_iters+1):

            loss_iter, loss_c_iter, kl_iter = func_update_model(self.mfdgp_handlers_objs.values(), self.mfdgp_handlers_cons.values(), optimizer)

            print("Iter:", i, "/", num_iters, ". Neg. ELBO per iter:", loss_iter.item())

            if i % self.step_plot == 0:

                self._plot_mfdgp_and_params(loss_c_iter, kl_iter, self.mfdgp_handlers_cons[ name_blackbox_to_plot ], loss_iter=loss_iter)

    def train_conditioned_mfdgps_and_plot(self, name_blackbox_to_plot, plt_show=False):

        def _update_conditioned_models(handlers_objs, handlers_cons, optimizer, plt_show=False):

            loss = 0.0
            loss_c = 0.0
            kl_c = 0.0
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
                    loss += -handler_objs.elbo(output, y_batch.T, fidelities, include_kl_term=False) / x_batch.shape[ 0 ] * handler_objs.num_data

                    output = handler_objs.mfdgp(self.pareto_set)
                    pareto_fidelities = torch.ones(size=(self.pareto_front.shape[ 0 ], 1)) * (handler_objs.num_fidelities - 1)
                    loss += -handler_objs.elbo(output, self.pareto_front[ :, i : (i + 1) ].T, pareto_fidelities, include_kl_term=False)

            for i, handler_cons in enumerate(handlers_cons):

                try:
                    (x_batch, y_batch, fidelities) = next(handler_cons.iter_train_loader)
                except:
                    handler_cons.iter_train_loader = iter(handler_cons.train_loader)
                    (x_batch, y_batch, fidelities) = next(handler_cons.iter_train_loader)

                with gpytorch.settings.num_likelihood_samples(1):
                    output = handler_cons.mfdgp(x_batch)
                    res = handler_cons.elbo(output, y_batch.T, fidelities)
                    loss_c += -res[0] / x_batch.shape[ 0 ] * handler_cons.num_data
                    kl_c += res[1]

                    # We add factors that ensure positive constraints at the pareto points. These are the theta factors.

                    output = handler_cons.mfdgp(self.pareto_set)[ handler_cons.num_fidelities - 1 ]
                    loss_c += -self.loss_theta_factors(output.mean, output.variance) 

                    loss += loss_c

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

            return loss, loss_c, kl_c

        self.loss_iters = self.loss_c_iters.copy()

        self._train_conditioned_mfdgps_and_plot(_update_conditioned_models, fix_variational_hypers=True, num_iters=self.num_epochs_1, lr=self.lr_1, name_blackbox_to_plot=name_blackbox_to_plot)
        # self._train_conditioned_mfdgps(_update_conditioned_models, fix_variational_hypers=False, num_iters=self.num_epochs_2, lr=self.lr_2)

        for handler_obj in self.mfdgp_handlers_objs.values():
            handler_obj.iter_train_loader = None

        for handler_con in self.mfdgp_handlers_cons.values():
            handler_con.iter_train_loader = None

        if plt_show: plt.show()        

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
    
    def get_model(self, name: str, is_constraint=False):

        if is_constraint:
            return self.mfdgp_handlers_cons[ name ].mfdgp
        
        return self.mfdgp_handlers_objs[ name ].mfdgp
    


def compute_moments_mfdgp(mfdgp, inputs, mean, std, fidelity):

    with gpytorch.settings.num_likelihood_samples(1):
        pred_means, pred_variances = mfdgp.predict_for_acquisition(inputs, fidelity)

    pred_mean = pred_means * std + mean
    pred_std  = np.sqrt(pred_variances) * std

    return pred_mean.numpy()[ 0 ], pred_std.numpy()[ 0 ]

def plot_model(ax,
               np_inputs,
               x_func0, x_func1,
               y_func0, y_func1,
               x_mf0, x_mf1,
               mean_mf0, mean_mf1,
               std_mf0, std_mf1,
               y_train_mf0, y_train_mf1,
               pred_mean_mf0, pred_mean_mf1,
               pred_std_mf0, pred_std_mf1,
               pareto_set=None, pareto_front_vals=None, cons=False):

    ax.plot(x_func0, y_func0, "b--", label="Low fidelity")
    ax.plot(x_func1, y_func1, "r--", label="High fidelity")

    if pareto_set is not None:
        assert pareto_front_vals is not None
        if cons:
            line, = ax.plot(pareto_set, pareto_front_vals*0.0, "b+")
            line.set_label('Loc pareto front')
        else:
            line, = ax.plot(pareto_set, pareto_front_vals * std_mf1 + mean_mf1, "b+")
            line.set_label('Pareto front')

    line, = ax.plot(x_mf0, y_train_mf0 * std_mf0 + mean_mf0, 'bX', markersize=12)
    line.set_label('Observed Data low fidelity')
    line, = ax.plot(x_mf1, y_train_mf1 * std_mf1 + mean_mf1, 'rX', markersize=12)
    line.set_label('Observed Data high fidelity')

    line, = ax.plot(np_inputs, pred_mean_mf1, 'g-')
    line.set_label('Predictive distribution MFDGP High Fidelity')
    line = ax.fill_between(np_inputs[:,0], (pred_mean_mf1 + pred_std_mf1), (pred_mean_mf1 - pred_std_mf1), color="green", alpha=0.5)
    line.set_label('Confidence MFDGP High Fidelity')

    line, = ax.plot(np_inputs, pred_mean_mf0, 'm-')
    line.set_label('Predictive distribution MFDGP Low Fidelity')
    line = ax.fill_between(np_inputs[:,0], (pred_mean_mf0 + pred_std_mf0), (pred_mean_mf0- pred_std_mf0), color="magenta", alpha=0.5)
    line.set_label('Confidence MFDGP Low Fidelity')

    ax.legend()
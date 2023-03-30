
import numpy as np
import torch
import gpytorch

from scipy.stats import norm
from torch.utils.data import TensorDataset, DataLoader

from mobocmf.models.mfdgp import MFDGP
from mobocmf.mlls.variational_elbo_mf import VariationalELBOMF
from mobocmf.util.moop import MOOP

dist = torch.distributions.normal.Normal(0, 1) # DFS: ¿Lo dejo asi? (No he encontrado una llamada a una funcion como en numpy solo un metodo de una clase)

# DFS: We could compare the performande of MFDGP-JESMOC with MESMOC+ if we set fidelities to 1,
# if we do that we could know how good is our approximation of the acquisition function

class MFDGPHandler():

    def __init__(self, x_train, y_train, fidelities_train, num_fidelities, batch_size):
        
        self.mfdgp = MFDGP(x_train, y_train, fidelities_train, num_fidelities=num_fidelities)
        self.mfdgp.double() # We use double precission to avoid numerical problems
        self.elbo = VariationalELBOMF(self.mfdgp, x_train.shape[-2], num_fidelities=num_fidelities)

        self.train_dataset = TensorDataset(x_train, y_train, fidelities_train)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

class BlackBoxMFDGPFitter():

    def __init__(self, num_fidelities, batch_size, lr_1=0.003, lr_2=0.001, num_epochs_1=5000, num_epochs_2=15000,
                 pareto_set_size=50, opt_grid_size=1000, eps=1e-8, decoupled_evals=False):
        
        assert decoupled_evals is False, "This object is not currently prepared for a decoupled evaluation setting."

        self.num_obj = 0
        self.num_con = 0

        self.mfdgp_handlers_objs = []
        self.mfdgp_handlers_cons = []

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

    def initialize_mfdgp(self, x_train, y_train, fidelities, threshold_constraint=0.0, is_constraint=False):

        if self.x_train is None:
            self.x_train = x_train
        else:
            assert torch.equal(self.x_train, x_train), "The inputs for this new mfdgp do not match with inputs for previous mfdgp models. This class is not currently prepared for a decoupled evaluation setting."
        
        if is_constraint:

            self.cons_train = torch.cat((self.cons_train, y_train), 1)

            self.mfdgp_handlers_cons.append(MFDGPHandler(x_train, y_train, fidelities, self.num_fidelities, self.batch_size))
            
            self.thresholds_cons = torch.cat((self.thresholds_cons, torch.tensor([threshold_constraint])), 0)

            self.num_con += 1

        else:

            self.objs_train = torch.cat((self.objs_train, y_train), 1)

            self.mfdgp_handlers_objs.append(MFDGPHandler(x_train, y_train, fidelities, self.num_fidelities, self.batch_size))

            self.num_obj += 1
    
    def _train_mfdgp(self, func_update_model, fix_variational_hypers, num_epochs, lr):

        l_opt_objs = []
        l_opt_cons = []

        for handler_obj in self.mfdgp_handlers_objs:

            handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)

            l_opt_objs.append(torch.optim.Adam([ {'params': handler_obj.mfdgp.parameters()} ], lr=lr))

        for handler_con in self.mfdgp_handlers_cons:

            handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)

            l_opt_cons.append(torch.optim.Adam([ {'params': handler_con.mfdgp.parameters()} ], lr=lr)) # Pasara qui los aprametros de todos los mdoelos (para el loss general)

        for i in range(num_epochs):
            
            for (handler_obj, optimizer) in zip(self.mfdgp_handlers_objs, l_opt_objs):
                
                loss_iter = func_update_model(handler_obj.mfdgp, handler_obj.elbo, optimizer, handler_obj.train_loader)

                print("[OBJ] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item())
            
            for (handler_con, optimizer) in zip(self.mfdgp_handlers_cons, l_opt_cons):
                
                loss_iter = func_update_model(handler_con.mfdgp, handler_con.elbo, optimizer, handler_con.train_loader)

                print("[CON] Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item())

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
        
    def get_pareto_solution(self):

        l_samples_objs = []

        for handler_obj in self.mfdgp_handlers_objs:
            l_samples_objs.append(handler_obj.mfdgp.sample_function_from_each_layer()[ -1 ])

        l_samples_cons = []
        
        for handler_con in self.mfdgp_handlers_cons:
            l_samples_cons.append(handler_con.mfdgp.sample_function_from_each_layer()[ -1 ])

        inputs = self.x_train
        # if self.decoupled_evals:
        #     for mfdgp in self.objs_mfdgp[ 1 : ]:
        #         inputs = np.unique(np.concatenate((inputs, mfdgp.x_train)), axis=1)
        #     
        #     for mfdgp in self.cons_mfdgp[ 1 : ]:
        #         inputs = np.unique(np.concatenate((inputs, mfdgp.x_train)), axis=1)

        global_optimizer = MOOP(l_samples_objs,
                                l_samples_cons,
                                input_dim=inputs.shape[ 1 ],
                                grid_size=self.opt_grid_size,
                                pareto_set_size=self.pareto_set_size)

        self.pareto_set, self.pareto_front = global_optimizer.compute_pareto_solution_from_samples(inputs)
        # self.pareto_set, self.pareto_front, self.pareto_front_cons = global_optimizer.compute_pareto_solution_from_samples(inputs)

        self.pareto_fidelities = torch.ones(size=(self.pareto_front.shape[ 0 ], 1)) * (self.num_fidelities - 1)

        return self.pareto_set, self.pareto_front #, self.pareto_front_cons

    def loss_theta_factors(self, cs_mean, cs_var):

        gamma_c_star = (cs_mean - self.thresholds_cons) / torch.sqrt(cs_var)

        cdf_gamma_c_star = dist.cdf(gamma_c_star)

        return torch.sum(np.log(1 - self.eps) * cdf_gamma_c_star + np.log(self.eps) * (1 - cdf_gamma_c_star))

    def loss_omega_factors(self, fs_mean, fs_var, cs_mean, cs_var, pareto_front):

        gamma_c      = (cs_mean - self.thresholds_cons) / torch.sqrt(cs_var)
        gamma_f_star = (pareto_front[ : , : , None] - fs_mean) / torch.sqrt(fs_var)

        cdf_gamma_c_times_cdf_gamma_f_star = torch.prod(dist.cdf(gamma_c), 1) * torch.prod(dist.cdf(gamma_f_star), 2)

        return torch.sum(np.log(1 - self.eps) * cdf_gamma_c_times_cdf_gamma_f_star
                         + np.log(self.eps) * (1 - cdf_gamma_c_times_cdf_gamma_f_star))


    def _train_conditioned_mfdgps(self, func_update_model, fix_variational_hypers, num_epochs, lr):

        params = list()

        for handler_obj in self.mfdgp_handlers_objs:
            
            handler_obj.mfdgp.fix_variational_hypers(fix_variational_hypers)
            params = params + list(handler_obj.mfdgp.parameters())

        for handler_con in self.mfdgp_handlers_cons:
            
            handler_con.mfdgp.fix_variational_hypers(fix_variational_hypers)
            params = params + list(handler_con.mfdgp.parameters())

        optimizer = torch.optim.Adam([ {'params': params } ], lr=lr)

        for i in range(num_epochs):

            loss_iter = func_update_model(self.mfdgp_handlers_objs, self.mfdgp_handlers_cons, optimizer)

            print("Epoch:", i, "/", num_epochs, ". Avg. Neg. ELBO per epoch:", loss_iter.item())

    def train_conditioned_mfdgps(self): # Proesar por separado los puntos de pareto (en todas las iteraciones se procesan)

        x_data = self.mfdgp_handlers_objs[ 0 ].train_dataset.tensors[ 0 ] + 0.0
        y_data = self.mfdgp_handlers_objs[ 0 ].train_dataset.tensors[ 1 ] + 0.0
        fidelities = self.mfdgp_handlers_objs[ 0 ].train_dataset.tensors[ 2 ] + 0.0

        for handler_objs in self.mfdgp_handlers_objs[ 1 : ]:

            y_data = torch.cat((y_data, handler_objs.train_dataset.tensors[ 1 ]), 1)

        for handler_cons in self.mfdgp_handlers_cons:

            y_data = torch.cat((y_data, handler_cons.train_dataset.tensors[ 1 ]), 1)
        
        self.train_dataset = TensorDataset(x_data, y_data, fidelities)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        def _update_conditioned_models(handlers_objs, handlers_cons, optimizer):

            loss_iter = 0.0
            
            for (x_batch, ys_batch, fidelities_batch) in self.train_loader:

                x_batch_and_pareto = torch.cat((self.pareto_set, x_batch), 0)
                fidelities_batch_and_pareto = torch.cat((self.pareto_fidelities, fidelities_batch), 0)

                # Samplear el x' de manera uniforme para cada minibatch
                x_tilde = torch.rand(size=((x_batch.shape[ 0 ], self.pareto_set.shape[ 1 ]))).double() # sampling: x_batch_pints * x_dim
                # x_tilde = torch.cat((x_tilde, x_batch[ ~m_pareto_pts, None ]), 0) # DFS: ¿Incluimos las observaciones en x_tilde o no?

                loss = 0.0
                optimizer.zero_grad()

                # Aqui deberia haber un bucle sobre los modelos para sumar al loss, y luego habria que sumar al loss los factores Omega
                with gpytorch.settings.num_likelihood_samples(1):

                    means_c_pareto = torch.tensor([], dtype=torch.double)
                    means_c_x_tilde = torch.tensor([], dtype=torch.double)
                    means_f_x_tilde = torch.tensor([], dtype=torch.double)
                    vars_c_pareto = torch.tensor([], dtype=torch.double)
                    vars_c_x_tilde = torch.tensor([], dtype=torch.double)
                    vars_f_x_tilde = torch.tensor([], dtype=torch.double)

                    for i, handler_objs in enumerate(handlers_objs):

                        y_batch_and_pareto = torch.cat((self.pareto_front[ None, : , i ], ys_batch[ None, : , i ]), 1)

                        output = handler_objs.mfdgp(x_batch_and_pareto) # We pass the data through the DGP to obtain the predictions
                        loss += -handler_objs.elbo(output, y_batch_and_pareto, fidelities_batch_and_pareto, num_data=y_batch_and_pareto.shape[ 1 ]) # / y_batch_and_pareto.shape[ 1 ]
                        
                        output_f_x_tilde = handler_objs.mfdgp(x_tilde)[ self.num_fidelities - 1 ]
                        mean_f_x_tilde, var_f_x_tilde = output_f_x_tilde.mean, output_f_x_tilde.variance

                        means_f_x_tilde = torch.cat((means_f_x_tilde, mean_f_x_tilde[None, :]), 0)
                        vars_f_x_tilde = torch.cat((vars_f_x_tilde, var_f_x_tilde[None, :]), 0)

                    for i, handler_cons in enumerate(handlers_cons):

                        y_batch = ys_batch[ None, : , i + self.num_obj ]

                        # We process x_batch points (observations and pareto set points)
                        output = handler_cons.mfdgp(x_batch) # We pass the data through the DGP to obtain the predictions
                        loss = -handler_cons.elbo(output, y_batch, fidelities_batch, num_data=y_batch.shape[ 1 ]) # / y_batch.shape[ 1 ]

                        # y_batch_and_pareto = torch.cat((self.pareto_front_cons[ None, : , i ], ys_batch[ None, : , i + self.num_obj ]), 1)
                        
                        # output = handler_cons.mfdgp(x_batch_and_pareto) # We pass the data through the DGP to obtain the predictions
                        # loss += -handler_cons.elbo(output, y_batch_and_pareto, fidelities_batch_and_pareto)

                        # We get the mean and variance for the theta factors 
                        output_c_pareto = handler_cons.mfdgp(self.pareto_set)[ self.num_fidelities - 1 ]
                        mean_c_pareto, var_c_pareto = output_c_pareto.mean, output_c_pareto.variance
                        means_c_pareto = torch.cat((means_c_pareto, mean_c_pareto[None, :]), 0)
                        vars_c_pareto = torch.cat((vars_c_pareto, var_c_pareto[None, :]), 0)

                        # We get the mean and variance for the omega factors
                        output_c_x_tilde = handler_cons.mfdgp(x_tilde)[ self.num_fidelities - 1 ] # We pass the data through the DGP to obtain the predictions
                        mean_c_x_tilde, var_c_x_tilde = output_c_x_tilde.mean, output_c_x_tilde.variance
                        means_c_x_tilde = torch.cat((means_c_x_tilde, mean_c_x_tilde[None, :]), 0)
                        vars_c_x_tilde = torch.cat((vars_c_x_tilde, var_c_x_tilde[None, :]), 0)

                    loss += -self.loss_theta_factors(means_c_pareto, vars_c_pareto) #/ means_c_pareto.shape[ 1 ] # Factores de las restricciones para que los puntos de la frontera cumplan las restricciones

                    loss += -self.loss_omega_factors(means_f_x_tilde, vars_f_x_tilde, means_c_x_tilde, vars_c_x_tilde, self.pareto_front) #/ means_f_x_tilde.shape[ 1 ]  # Factores omega para que el resto de puntos del espacio sean ajustados para que tengan coherencia con la frontera
                    
                    loss.backward()
                    optimizer.step()
                    loss_iter += loss

            return loss_iter

        self._train_conditioned_mfdgps(_update_conditioned_models, fix_variational_hypers=True, num_epochs=self.num_epochs_1, lr=self.lr_1)
        self._train_conditioned_mfdgps(_update_conditioned_models, fix_variational_hypers=False, num_epochs=self.num_epochs_2, lr=self.lr_2)



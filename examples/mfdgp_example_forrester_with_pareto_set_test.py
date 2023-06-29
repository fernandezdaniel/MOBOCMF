import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from mobocmf.models.exact_gp import ExactGPModel
from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.models.mfdgp import MFDGP
from mobocmf.mlls.variational_elbo_mf import VariationalELBOMF

# XXX DFS: We use Walrus Operator so the python version should be >= 3.8
import sys; assert sys.version_info[1] >= 8

# We obtain the high fidelity dataset and dataloader

np.random.seed(0)

num_inputs_high_fidelity = 4
num_inputs_low_fidelity = 12

num_epochs_1 = 2
num_epochs_2 = 4
batch_size = num_inputs_low_fidelity + num_inputs_high_fidelity

lower_limit = 0.0
upper_limit = 1.0

func_obj1_mf0 = forrester_mf0
func_obj1_mf1 = forrester_mf1

func_con1_mf0 = lambda x: np.sin(x * np.pi * 2.5) # non_linear_sin_mf0
func_con1_mf1 = lambda x: np.cos(x * np.pi * 2.5) # non_linear_sin_mf1

# We obtain x and ys for the low fidelity 

x_mf0 = np.linspace(0, 1.0, num_inputs_low_fidelity).reshape((num_inputs_low_fidelity, 1))
obj1_mf0 = func_obj1_mf0(x_mf0)
obj2_mf0 = func_obj1_mf0(x_mf0) * -1.0
con1_mf0 = func_con1_mf0(x_mf0)

# We obtain x and ys for the high fidelity 

upper_limit_high_fidelity = (upper_limit - lower_limit) * 0.7 + lower_limit
x_mf1 = np.array([0.1, 0.3, 0.5, 0.7]).reshape((num_inputs_high_fidelity, 1))
obj1_mf1 = func_obj1_mf1(x_mf1)
obj2_mf1 = func_obj1_mf1(x_mf1) * -1.0
con1_mf1 = func_con1_mf1(x_mf1)

def preprocess_outputs(y_low, y_high):

    # Important: DHL we use the same mean and standar deviation for each fidelity !!!

    y_high_mean = np.mean(np.vstack((y_high, y_low)))
    y_high_std = np.std(np.vstack((y_high, y_low)))
    y_low_mean = y_high_mean
    y_low_std = y_high_std

    y_high = (y_high - y_high_mean) / y_high_std
    y_train_high = torch.from_numpy(y_high).double()

    y_low = (y_low - y_low_mean) / y_low_std
    y_train_low = torch.from_numpy(y_low).double()

    return y_train_low, y_train_high, y_low_mean, y_high_mean, y_low_std, y_high_std

obj1_train_mf0, obj1_train_mf1, obj1_mean_mf0, obj1_std_mf0, obj1_mean_mf1, obj1_std_mf1 = \
    preprocess_outputs(obj1_mf0, obj1_mf1)

obj2_train_mf0, obj2_train_mf1, obj2_mean_mf0, obj2_std_mf0, obj2_mean_mf1, obj2_std_mf1 = \
    preprocess_outputs(obj2_mf0, obj2_mf1)

con1_train_mf0, con1_train_mf1, con1_mean_mf0, con1_std_mf0, con1_mean_mf1, con1_std_mf1 = \
    preprocess_outputs(con1_mf0, con1_mf1)

x_train_mf0 = torch.from_numpy(x_mf0).double()
x_train_mf1 = torch.from_numpy(x_mf1).double()

# We obtain x and y for all fidelities

# x_train = torch.cat((x_train_mf1, x_train_mf0), 0)

x_train = torch.cat((torch.cat((x_train_mf1, x_train_mf0), 0),
                     torch.cat((x_train_mf1, x_train_mf0), 0)*0.5 # XXX DFS: We add a fake second dimension only for the test
                    ), 1)
obj1_train = torch.cat((obj1_train_mf1, obj1_train_mf0), 0)
obj2_train = torch.cat((obj2_train_mf1, obj2_train_mf0), 0)
con1_train = torch.cat((con1_train_mf1, con1_train_mf0), 0)
fid = torch.cat((torch.ones(len(x_mf1)).double(), torch.zeros(len(x_mf0)).double()))[:, None] 

obj1_train_dataset = TensorDataset(x_train, obj1_train, fid)
obj1_train_loader  = DataLoader(obj1_train_dataset, batch_size=batch_size, shuffle=True)

obj2_train_dataset = TensorDataset(x_train, obj2_train, fid)
obj2_train_loader  = DataLoader(obj2_train_dataset, batch_size=batch_size, shuffle=True)

con1_train_dataset = TensorDataset(x_train, con1_train, fid)
con1_train_loader  = DataLoader(con1_train_dataset, batch_size=batch_size, shuffle=True)


# We create the objects for the mfdgp model and the approximate marginal log likelihood of the mfdgp

def create_mfdgp_model(x_train, y_train, fid, num_fidelities):
    model = MFDGP(x_train, y_train, fid, num_fidelities=num_fidelities)
    model.double() # We use double precission to avoid numerical problems
    elbo = VariationalELBOMF(model, x_train.shape[-2], num_fidelities=num_fidelities)

    model.fix_variational_hypers(True)

    return model, elbo

mfdgp_obj1, elbo_obj1 = create_mfdgp_model(x_train, obj1_train, fid, 2)
mfdgp_obj2, elbo_obj2 = create_mfdgp_model(x_train, obj2_train, fid, 2)
mfdgp_con1, elbo_con1 = create_mfdgp_model(x_train, con1_train, fid, 2)

def train_mfdgp_model(model, elbo, train_loader, num_epochs_1, num_epochs_2):
    optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=0.003) 

    for i in range(num_epochs_1):

        minibatch_iter = train_loader
        loss_iter = 0.0
        kl_iter = 0.0

        for (x_batch, y_batch, fidelities) in minibatch_iter:

            with gpytorch.settings.num_likelihood_samples(1):
                optimizer.zero_grad() 
                output = model(x_batch)
                loss, kl = -elbo(output, y_batch.T, fidelities)
                loss.backward()                
                optimizer.step() 
                loss_iter += loss
                kl_iter += kl

        print("Epoch:", i, "/", num_epochs_1, ". Avg. Neg. ELBO per epoch:", loss_iter.item(), "\t KL per epoch:", kl_iter.item())

    model.fix_variational_hypers(False)

    optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=0.001)

    for i in range(num_epochs_2):

        minibatch_iter = train_loader
        loss_iter = 0.0

        for (x_batch, y_batch, fidelities) in minibatch_iter:

            with gpytorch.settings.num_likelihood_samples(1):
                optimizer.zero_grad() 
                output = model(x_batch)
                loss = -elbo(output, y_batch.T, fidelities)
                loss.backward()                
                optimizer.step() 
                loss_iter += loss

        print("Epoch:", i, "/", num_epochs_2, ". Avg. Neg. ELBO per epoch:", loss_iter.item())

    return model

mfdgp_obj1 = train_mfdgp_model(mfdgp_obj1, elbo_obj1, obj1_train_loader, num_epochs_1, num_epochs_2)
mfdgp_obj2 = train_mfdgp_model(mfdgp_obj2, elbo_obj2, obj2_train_loader, num_epochs_1, num_epochs_2)
mfdgp_con1 = train_mfdgp_model(mfdgp_con1, elbo_con1, con1_train_loader, num_epochs_1, num_epochs_2)

# def compute_moments_mfdgp(model, y_high_mean, y_high_std, lower_limit, upper_limit, num_fidelity=0, num_samples=100):
#     model.eval()

#     test_inputs  = (torch.linspace(lower_limit, upper_limit, steps=30)).double()
#     test_inputs  = torch.cartesian_prod(test_inputs, test_inputs)

#     samples_high = np.zeros((num_samples, test_inputs.shape[ 0 ]))

#     for i in range(num_samples):
#         with gpytorch.settings.num_likelihood_samples(1):
#             pred_means, pred_variances = model.predict(test_inputs, num_fidelity)
#             samples_high[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
#                     np.sqrt(pred_variances.numpy()) + pred_means.numpy()

#     pred_mean_high = np.mean(samples_high, 0) * y_high_std + y_high_mean
#     pred_std_high  = np.std(samples_high, 0) * y_high_std

#     return pred_mean_high, pred_std_high

# obj1_mf1_mean, _ = compute_moments_mfdgp(mfdgp_obj1, obj1_mean_mf1, obj1_std_mf1, lower_limit, upper_limit, num_fidelity=1)
# obj2_mf1_mean, _ = compute_moments_mfdgp(mfdgp_obj2, obj2_mean_mf1, obj2_std_mf1, lower_limit, upper_limit, num_fidelity=1)
# con1_mf1_mean, _ = compute_moments_mfdgp(mfdgp_con1, con1_mean_mf1, con1_std_mf1, lower_limit, upper_limit, num_fidelity=1)

# We sample from the MFDG(five samples are generated from each fidelity)

# def sample_solution(num_dims, obj_model_dict, con_models_dict):

# samples_obj1_mf1 = []
# samples_obj2_mf1 = []
# samples_con1_mf1 = []
# for i in range(5):
#     samples_obj1_mf1.append(mfdgp_obj1.sample_function_from_each_layer()[1])
#     samples_obj2_mf1.append(mfdgp_obj2.sample_function_from_each_layer()[1])
#     samples_con1_mf1.append(mfdgp_con1.sample_function_from_each_layer()[1])


from mobocmf.util.moop import MOOP

# Para saber si funciona bien la construccion del conjunto de pareto ejecutar las siguientes lineas

samples_obj1 = mfdgp_obj1.sample_function_from_each_layer()
samples_obj2 = mfdgp_obj2.sample_function_from_each_layer()
samples_con1 = mfdgp_con1.sample_function_from_each_layer()

# global_optimizer = MOOP([samples_obj1[0], samples_obj2[0]],
#                         [samples_con1[0]],
#                         input_dim=2,
#                         grid_size=1000,
#                         pareto_set_size=50)


# pareto_set, pareto_front = global_optimizer.compute_pareto_solution_from_samples()

# Para comprobar que funciona bien lo del gradiente en la segunda capa ejecutar las siguientes lineas:

global_optimizer = MOOP([samples_obj1[1], samples_obj2[1]],
                        [samples_con1[1]],
                        input_dim=2,
                        grid_size=1000,
                        pareto_set_size=50) # plt.plot(pareto_front[:, 0], pareto_front[:, 1], "+")


pareto_set, pareto_front = global_optimizer.compute_pareto_solution_from_samples(x_train_mf1)

plt.plot(pareto_set[:, 0], pareto_set[:, 1], "Xk", label="Pareto set")
plt.legend()
plt.title("Pareto set")
plt.show()

import pdb; pdb.set_trace()

# def obj1_mf1_mean(x, num_samples=30):
#     with gpytorch.settings.num_likelihood_samples(1):
#         res = mfdgp_obj1.predict(x, 1)[0]
#     for _ in range(1, num_samples):
#         with gpytorch.settings.num_likelihood_samples(1):
#             res += mfdgp_obj1.predict(x, 1)[0]
#     return res / num_samples
# def obj2_mf1_mean(x, num_samples=30):
#     with gpytorch.settings.num_likelihood_samples(1):
#         res = mfdgp_obj2.predict(x, 1)[0]
#     for _ in range(1, num_samples):
#         with gpytorch.settings.num_likelihood_samples(1):
#             res += mfdgp_obj2.predict(x, 1)[0]
#     return res / num_samples
# def con1_mf1_mean(x, num_samples=30):
#     with gpytorch.settings.num_likelihood_samples(1):
#         res = mfdgp_con1.predict(x, 1)[0]
#     for _ in range(1, num_samples):
#         with gpytorch.settings.num_likelihood_samples(1):
#             res += mfdgp_con1.predict(x, 1)[0]
#     return res / num_samples

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

import dill as pickle

from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.models.mfdgp import TL
from mobocmf.util.blackbox_mfdgp_fitter_plot import BlackBoxMFDGPFitterPlot
from mobocmf.acquisition_functions.JESMOC_MFDGP import JESMOC_MFDGP

from mobocmf.util.util import create_path, save_pickle, read_pickle

input_dims = 1

high_fidelity = forrester_mf1
low_fidelity  = forrester_mf0

# We obtain the high fidelity dataset and dataloader

np.random.seed(0)

num_fidelities = 2

num_inputs_high_fidelity = 4
num_inputs_low_fidelity = 8

num_epochs_1 = 5000
num_epochs_2 = 15000
batch_size = num_inputs_low_fidelity + num_inputs_high_fidelity # DFS: Cambiar para ver que pasa cuando el batch no es del numero de datos

lower_limit = 0.0
upper_limit = 1.0

func_obj1_mf0 = forrester_mf0
func_obj1_mf1 = forrester_mf1

func_obj2_mf0 = lambda x: forrester_mf0(x) * -1.0
func_obj2_mf1 = lambda x: forrester_mf1(x) * -1.0

func_con1_mf0 = lambda x: np.sin(x * np.pi * 2.5) # non_linear_sin_mf0
func_con1_mf1 = lambda x: np.cos(x * np.pi * 2.5) # non_linear_sin_mf1

# We obtain x and ys for the low fidelity 

x_mf0 = np.linspace(0, 1.0, num_inputs_low_fidelity).reshape((num_inputs_low_fidelity, 1))
obj1_mf0 = func_obj1_mf0(x_mf0)
obj2_mf0 = func_obj2_mf0(x_mf0)
con1_mf0 = func_con1_mf0(x_mf0)

# We obtain x and ys for the high fidelity 

upper_limit_high_fidelity = (upper_limit - lower_limit) * 0.7 + lower_limit
x_mf1 = np.array([0.1, 0.3, 0.5, 0.7]).reshape((num_inputs_high_fidelity, 1))
obj1_mf1 = func_obj1_mf1(x_mf1)
obj2_mf1 = func_obj2_mf1(x_mf1)
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
    
obj1_train_mf0, obj1_train_mf1, obj1_mean_mf0, obj1_mean_mf1, obj1_std_mf0, obj1_std_mf1 = \
    preprocess_outputs(obj1_mf0, obj1_mf1)

obj2_train_mf0, obj2_train_mf1, obj2_mean_mf0, obj2_mean_mf1, obj2_std_mf0, obj2_std_mf1 = \
    preprocess_outputs(obj2_mf0, obj2_mf1)

con1_train_mf0, con1_train_mf1, con1_mean_mf0, con1_mean_mf1, con1_std_mf0, con1_std_mf1 = \
    preprocess_outputs(con1_mf0, con1_mf1)

threshold_constraint = 0.0 * con1_std_mf1 + con1_mean_mf1

x_train_mf0 = torch.from_numpy(x_mf0).double()
x_train_mf1 = torch.from_numpy(x_mf1).double()

# We obtain x and y for all fidelities

x_train = torch.cat((x_train_mf1, x_train_mf0), 0)
# x_train = torch.cat((torch.cat((x_train_mf1, x_train_mf0), 0),
#                      torch.cat((x_train_mf1, x_train_mf0), 0)*0.5 # XXX DFS: We add a fake second dimension only for the test
#                     ), 1)
obj1_train = torch.cat((obj1_train_mf1, obj1_train_mf0), 0)
obj2_train = torch.cat((obj2_train_mf1, obj2_train_mf0), 0)
con1_train = torch.cat((con1_train_mf1, con1_train_mf0), 0)
fidelities = torch.cat((torch.ones(len(x_mf1)).double(), torch.zeros(len(x_mf0)).double()))[ : , None ]

blackbox_mfdgp_fitter = BlackBoxMFDGPFitterPlot(num_fidelities, batch_size,
                                                num_epochs_1=num_epochs_1, num_epochs_2=num_epochs_2,
                                                type_lengthscale=TL.MEDIAN, step_plot=50)

blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj1_train, fidelities, "obj1", is_constraint=False)
blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj2_train, fidelities, "obj2", is_constraint=False)
blackbox_mfdgp_fitter.initialize_mfdgp(x_train, con1_train, fidelities, "con1", threshold_constraint=threshold_constraint, is_constraint=True)

##################################################################################################
# Unconditioned training

blackbox_mfdgp_fitter._set_parameters_to_plot_con1(lower_limit,    upper_limit,
                                                  func_con1_mf0,  func_con1_mf1,
                                                  x_mf0, x_mf1,
                                                  con1_mean_mf0,  con1_mean_mf1,
                                                  con1_std_mf0,   con1_std_mf1,
                                                  con1_train_mf0, con1_train_mf1)

blackbox_mfdgp_fitter.train_mfdgps_and_plot(name_blackbox_to_plot='con1', plt_show=False)

# filename = "mfdgp_uncond_%dxmf0_%dxmf1_%d_%d_new_23062023.dat" % (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_1, num_epochs_2)
# save_pickle("./blackbox_mfdgp_fitters/", filename, blackbox_mfdgp_fitter)
# blackbox_mfdgp_fitter = read_pickle("./blackbox_mfdgp_fitters/", filename)

##################################################################################################
##################################################################################################


##################################################################################################
# Conditioned training

blackbox_mfdgp_fitter.num_epochs_1 = 15000
blackbox_mfdgp_fitter.num_epochs_2 = 0

blackbox_mfdgp_fitter.sample_and_store_pareto_solution()

blackbox_mfdgp_fitter.train_conditioned_mfdgps_and_plot(name_blackbox_to_plot='con1', plt_show=True)

filename = "mfdgp_cond_%dxmf0_%dxmf1_%d_%d_new_23062023.dat" % (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_1, num_epochs_2)
save_pickle("./blackbox_mfdgp_fitters/", filename, blackbox_mfdgp_fitter)
blackbox_mfdgp_fitter = read_pickle("./blackbox_mfdgp_fitters/", filename)

##################################################################################################
##################################################################################################


import pdb; pdb.set_trace()


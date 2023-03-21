import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.util.blackbox_mfdgp_fitter import BlackBoxMFDGPFitter

# XXX DFS: We use Walrus Operator so the python version should be >= 3.8
import sys; assert sys.version_info[1] >= 8

input_dims = 1

high_fidelity = forrester_mf1
low_fidelity  = forrester_mf0

# We obtain the high fidelity dataset and dataloader

np.random.seed(0)

num_fidelities = 2

num_inputs_high_fidelity = 4
num_inputs_low_fidelity = 12

num_epochs_1 = 1000
num_epochs_2 = 3000
batch_size = num_inputs_low_fidelity + num_inputs_high_fidelity

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

blackbox_mfdgp_fitter = BlackBoxMFDGPFitter(num_fidelities, batch_size, num_epochs_1=num_epochs_1, num_epochs_2=num_epochs_2)

blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj1_train, fidelities, is_constraint=False)
blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj2_train, fidelities, is_constraint=False)
blackbox_mfdgp_fitter.initialize_mfdgp(x_train, con1_train, fidelities, is_constraint=True)


##########################################################################################################
# Unconditioned training

# blackbox_mfdgp_fitter.train_mfdgps()

def compute_moments_mfdgp(mfdgp, inputs, mean, std, fidelity, num_samples=100):

    samples = np.zeros((num_samples, inputs.shape[ 0 ]))

    # with gpytorch.settings.num_likelihood_samples(1): 
    #     ret = mfdgp.hidden_layer_1(torch.from_numpy(np.hstack((spacing[:,0:1], spacing[:,0:1]))))

    for i in range(num_samples):
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means, pred_variances = mfdgp.predict(inputs, fidelity)
            samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                    np.sqrt(pred_variances.numpy()) + pred_means.numpy()

    pred_mean = np.mean(samples, 0) * std + mean
    pred_std  = np.std(samples, 0) * std

    return pred_mean, pred_std

def plot_black_box(inputs,
                   func_mf0, func_mf1,
                   x_mf0, x_mf1,
                   mean_mf0, mean_mf1,
                   std_mf0, std_mf1,
                   y_train_mf0, y_train_mf1,
                   pred_mean_mf0, pred_mean_mf1,
                   pred_std_mf0, pred_std_mf1,
                   lower_limit, upper_limit,
                   pareto_set=None, pareto_front_vals=None, cons=False):
                   
    # We plot the model

    spacing = np.linspace(lower_limit, upper_limit, 1000)[:, None]
    _, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(spacing, func_mf0(spacing), "b--", label="Low fidelity")
    ax.plot(spacing, func_mf1(spacing), "r--", label="High fidelity")

    if pareto_set is not None:
        if cons:
            ax.plot(pareto_set, pareto_front_vals * 0.0, "+", label="Loc pareto front")
        else:
            ax.plot(pareto_set, pareto_front_vals * std_mf1 + mean_mf1, "+", label="Pareto front")

    line, = ax.plot(x_mf0, y_train_mf0 * std_mf0 + mean_mf0, 'bX', markersize=12)
    line.set_label('Observed Data low fidelity')
    line, = ax.plot(x_mf1, y_train_mf1 * std_mf1 + mean_mf1, 'rX', markersize=12)
    line.set_label('Observed Data high fidelity')

    line, = ax.plot(inputs.numpy(), pred_mean_mf1, 'g-')
    line.set_label('Predictive distribution MFDGP High Fidelity')
    line = ax.fill_between(inputs.numpy()[:,0], (pred_mean_mf1 + pred_std_mf1), (pred_mean_mf1 - pred_std_mf1), color="green", alpha=0.5)
    line.set_label('Confidence MFDGP High Fidelity')

    line, = ax.plot(inputs.numpy(), pred_mean_mf0, 'm-')
    line.set_label('Predictive distribution MFDGP Low Fidelity')
    line = ax.fill_between(inputs.numpy()[:,0], (pred_mean_mf0 + pred_std_mf0), (pred_mean_mf0- pred_std_mf0), color="magenta", alpha=0.5)
    line.set_label('Confidence MFDGP Low Fidelity')

    plt.legend()
    plt.show()

NUM_SAMPLES = 10

spacing = torch.linspace(lower_limit, upper_limit, 200).double()[ : , None ] # torch.rand(size=(1000,)).double() # torch.rand(size=(1000, input_dims)).double() # Samplear de manera unif tal como se hacia en Spearmint y hacerlo para cada batch

# mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ 0 ].mfdgp
# mfdgp.eval()

# obj1_pred_mean_mf0, obj1_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               obj1_mean_mf0, obj1_std_mf0,
#                                                               fidelity=0, num_samples=NUM_SAMPLES)
# obj1_pred_mean_mf1, obj1_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               obj1_mean_mf1, obj1_std_mf1,
#                                                               fidelity=1, num_samples=NUM_SAMPLES)
# mfdgp.train()

# plot_black_box(spacing,
#                func_obj1_mf0,  func_obj1_mf1,
#                x_mf0, x_mf1,
#                obj1_mean_mf0,  obj1_mean_mf1,
#                obj1_std_mf0,   obj1_std_mf1,
#                obj1_train_mf0, obj1_train_mf1,
#                obj1_pred_mean_mf0, obj1_pred_mean_mf1,
#                obj1_pred_std_mf0,  obj1_pred_std_mf1,
#                lower_limit, upper_limit)


# mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ 1 ].mfdgp
# mfdgp.eval()

# obj2_pred_mean_mf0, obj2_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               obj2_mean_mf0, obj2_std_mf0,
#                                                               fidelity=0, num_samples=NUM_SAMPLES)
# obj2_pred_mean_mf1, obj2_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               obj2_mean_mf1, obj2_std_mf1,
#                                                               fidelity=1, num_samples=NUM_SAMPLES)
# mfdgp.train()

# plot_black_box(spacing,
#                func_obj2_mf0,  func_obj2_mf1,
#                x_mf0, x_mf1,
#                obj2_mean_mf0,  obj2_mean_mf1,
#                obj2_std_mf0,   obj2_std_mf1,
#                obj2_train_mf0, obj2_train_mf1,
#                obj2_pred_mean_mf0, obj2_pred_mean_mf1,
#                obj2_pred_std_mf0,  obj2_pred_std_mf1,
#                lower_limit, upper_limit)


# mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_cons[ 0 ].mfdgp
# mfdgp.eval()

# con1_pred_mean_mf0, con1_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               con1_mean_mf0, con1_std_mf0,
#                                                               fidelity=0, num_samples=NUM_SAMPLES)
# con1_pred_mean_mf1, con1_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
#                                                               con1_mean_mf1, con1_std_mf1,
#                                                               fidelity=1, num_samples=NUM_SAMPLES)
# mfdgp.train()

# plot_black_box(spacing,
#                func_con1_mf0,  func_con1_mf1,
#                x_mf0, x_mf1,
#                con1_mean_mf0,  con1_mean_mf1,
#                con1_std_mf0,   con1_std_mf1,
#                con1_train_mf0, con1_train_mf1,
#                con1_pred_mean_mf0, con1_pred_mean_mf1,
#                con1_pred_std_mf0,  con1_pred_std_mf1,
#                lower_limit, upper_limit)

# import copy
# blackbox_mfdgp_fitter_uncond = copy.deepcopy(blackbox_mfdgp_fitter)

##########################################################################################################
# Conditioned training

import dill as pickle

# with open("blackbox_mfdgp_fitter_uncond.dat", "wb") as fw:
#     pickle.dump(blackbox_mfdgp_fitter, fw)

with open("blackbox_mfdgp_fitter_uncond.dat", "rb") as fr:
    blackbox_mfdgp_fitter = pickle.load(fr)

pareto_set, pareto_front = blackbox_mfdgp_fitter.get_pareto_solution()


blackbox_mfdgp_fitter.train_conditioned_mfdgps() #, l_objs_pred_means, l_cons_pred_means) #, l_objs_pred_stds, l_cons_pred_stds)




mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ 0 ].mfdgp
mfdgp.eval()

obj1_pred_mean_mf0, obj1_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
                                                              obj1_mean_mf0, obj1_std_mf0,
                                                              fidelity=0, num_samples=NUM_SAMPLES)
obj1_pred_mean_mf1, obj1_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
                                                              obj1_mean_mf1, obj1_std_mf1,
                                                              fidelity=1, num_samples=NUM_SAMPLES)
mfdgp.train()

plot_black_box(spacing,
               func_obj1_mf0,  func_obj1_mf1,
               x_mf0, x_mf1,
               obj1_mean_mf0,  obj1_mean_mf1,
               obj1_std_mf0,   obj1_std_mf1,
               obj1_train_mf0, obj1_train_mf1,
               obj1_pred_mean_mf0, obj1_pred_mean_mf1,
               obj1_pred_std_mf0,  obj1_pred_std_mf1,
               lower_limit, upper_limit,
               pareto_set, pareto_front_vals=pareto_front[ : , 0 ])


mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ 1 ].mfdgp
mfdgp.eval()

obj2_pred_mean_mf0, obj2_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
                                                              obj2_mean_mf0, obj2_std_mf0,
                                                              fidelity=0, num_samples=NUM_SAMPLES)
obj2_pred_mean_mf1, obj2_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
                                                              obj2_mean_mf1, obj2_std_mf1,
                                                              fidelity=1, num_samples=NUM_SAMPLES)
mfdgp.train()

plot_black_box(spacing,
               func_obj2_mf0,  func_obj2_mf1,
               x_mf0, x_mf1,
               obj2_mean_mf0,  obj2_mean_mf1,
               obj2_std_mf0,   obj2_std_mf1,
               obj2_train_mf0, obj2_train_mf1,
               obj2_pred_mean_mf0, obj2_pred_mean_mf1,
               obj2_pred_std_mf0,  obj2_pred_std_mf1,
               lower_limit, upper_limit,
               pareto_set=pareto_set, pareto_front_vals=pareto_front[ : , 1 ])


mfdgp = blackbox_mfdgp_fitter.mfdgp_handlers_cons[ 0 ].mfdgp
mfdgp.eval()

con1_pred_mean_mf0, con1_pred_std_mf0 = compute_moments_mfdgp(mfdgp, spacing,
                                                              con1_mean_mf0, con1_std_mf0,
                                                              fidelity=0, num_samples=NUM_SAMPLES)
con1_pred_mean_mf1, con1_pred_std_mf1 = compute_moments_mfdgp(mfdgp, spacing,
                                                              con1_mean_mf1, con1_std_mf1,
                                                              fidelity=1, num_samples=NUM_SAMPLES)
mfdgp.train()

plot_black_box(spacing,
               func_con1_mf0,  func_con1_mf1,
               x_mf0, x_mf1,
               con1_mean_mf0,  con1_mean_mf1,
               con1_std_mf0,   con1_std_mf1,
               con1_train_mf0, con1_train_mf1,
               con1_pred_mean_mf0, con1_pred_mean_mf1,
               con1_pred_std_mf0,  con1_pred_std_mf1,
               lower_limit, upper_limit,
               pareto_set=pareto_set, pareto_front_vals=pareto_front[ : , 1 ], cons=True)

import pdb; pdb.set_trace()

# La distrib_cond debe ser compatible con los datos observados, la frontera y cumplir las cons

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from copy import deepcopy
from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.util.blackbox_mfdgp_fitter import BlackBoxMFDGPFitter
from mobocmf.acquisition_functions.JESMOC_MFDGP import JESMOC_MFDGP
from mobocmf.models.mfdgp import MFDGP

# XXX DFS: We use Walrus Operator so the python version should be >= 3.8
import sys; assert sys.version_info[1] >= 8
import dill as pickle
from botorch.optim.optimize import optimize_acqf

SEED = 2
NUM_BO_ITERS = 100
COST_HIGHER_FIDELITY=10
np.random.seed(SEED)

numpy_grid_for_optimal_hv = np.random.uniform(size = ((1000 * 2, 2)))

# We generate fake data to build a model to sample from the prior

x_train = torch.from_numpy(np.random.normal(size = ((10, 2))))
y_train = torch.from_numpy(np.random.normal(size = ((10, 1))))
fidelities = np.ones((10,1))
fidelities[ 0 : 5 ] = 0
fidelities = torch.from_numpy(fidelities)

# First the first objective

obj1 = MFDGP(x_train, y_train, fidelities, 2)
low_fidelity_obj1, high_fidelity_obj1 = obj1.sample_function_from_prior_each_layer()
#low_fidelity_obj1 = high_fidelity_obj1

obj2 = MFDGP(x_train, y_train, fidelities, 2)
low_fidelity_obj2, high_fidelity_obj2 = obj2.sample_function_from_prior_each_layer()
#low_fidelity_obj2 = high_fidelity_obj2

con1 = MFDGP(x_train, y_train, fidelities, 2)
low_fidelity_con1, high_fidelity_con1 = con1.sample_function_from_prior_each_layer()
#low_fidelity_con1 = high_fidelity_con1

con2 = MFDGP(x_train, y_train, fidelities, 2)
low_fidelity_con2, high_fidelity_con2 = con2.sample_function_from_prior_each_layer()
#low_fidelity_con2 = high_fidelity_con2

# We obtain the high fidelity dataset 

num_fidelities = 2

num_inputs_high_fidelity = 5
num_inputs_low_fidelity = 10

num_epochs_1 = 5
num_epochs_2 = 15

lower_limit = 0.0
upper_limit = 1.0

def preprocess_outputs(y_low, y_high):

    # Important: DHL we use the same mean and standar deviation for each fidelity !!!

    y_mean = np.mean(np.vstack((y_high, y_low)))
    y_std = np.std(np.vstack((y_high, y_low)))

    y_high = (y_high - y_mean) / y_std
    y_train_high = torch.from_numpy(y_high).double()

    y_low = (y_low - y_mean) / y_std
    y_train_low = torch.from_numpy(y_low).double()

    return y_train_low, y_train_high, y_mean, y_std

def compute_moments_mfdgp_for_acquisition(mfdgp, inputs, mean, std, fidelity, num_samples = None):

    with gpytorch.settings.num_likelihood_samples(1):
        with torch.no_grad():
            pred_means, pred_variances = mfdgp.predict_for_acquisition(inputs, fidelity)

    pred_mean = pred_means * std + mean
    pred_variance = pred_variances * std**2

    return pred_mean, torch.sqrt(pred_variance)

def compute_moments_mfdgp(mfdgp, inputs, mean, std, fidelity, num_samples=1000):

    samples = np.zeros((num_samples, inputs.shape[ 0 ]))

    for i in range(num_samples):
        with gpytorch.settings.num_likelihood_samples(1):
            with torch.no_grad():
                pred_means, pred_variances = mfdgp.predict(inputs, fidelity)
                samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                    np.sqrt(pred_variances.numpy()) + pred_means.numpy()

    pred_mean = np.mean(samples, 0) * std + mean
    pred_std  = np.std(samples, 0) * std

    return pred_mean, pred_std

def plot_black_box(mfdgp, func_mf0, func_mf1, x_train, y_train, fidelities, y_mean, y_std, name = "", filename = "", \
                   pareto_set=None, sampled_black_box = None, NUM_SAMPLES = 100):

    # First we compute the grid to plot the predictions

    x1 = np.linspace(0, 1, 25).reshape((25, 1))
    x2 = np.linspace(0, 1, 25).reshape((25, 1))
    xx, yy = np.meshgrid(x1, x2)
    grid = torch.from_numpy(np.vstack((xx.flatten(), yy.flatten())).T)
    numpy_grid = grid.numpy()

    pred_mean_mf0, pred_std_mf0 = compute_moments_mfdgp(mfdgp, grid, y_mean, y_std, fidelity=0, num_samples=NUM_SAMPLES)
    pred_mean_mf1, pred_std_mf1 = compute_moments_mfdgp(mfdgp, grid, y_mean, y_std, fidelity=1, num_samples=NUM_SAMPLES)

    true_low_fidelity_values = func_mf0(numpy_grid)
    true_high_fidelity_values = func_mf1(numpy_grid)

    # We plot data for each fidelity

    if pareto_set is not None:
        fig, ax = plt.subplots(4,2)
    else:
        fig, ax = plt.subplots(3,2)

    ax[ 0, 0 ].plot(x_train[ fidelities[ :, 0 ] == 0, 0 ], x_train[ fidelities[ :, 0 ] == 0, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 0, 0 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 0, 0 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , true_low_fidelity_values.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 0, 0 ].set_title("True Low Fidelity " + name, fontsize = 8)

    ax[ 0, 1 ].plot(x_train[ fidelities[ :, 0 ] == 1, 0 ], x_train[ fidelities[ :, 0 ] == 1, 1 ], color='red', marker='o', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 0, 1 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 0, 1 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , true_high_fidelity_values.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 0, 1 ].set_title("True High Fidelity " + name, fontsize = 8)
 
    ax[ 1, 0 ].plot(x_train[ fidelities[ :, 0 ] == 0, 0 ], x_train[ fidelities[ :, 0 ] == 0, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 1, 0 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 1, 0 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , pred_mean_mf0.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 1, 0 ].set_title("Pred Mean Low Fidelity " + name, fontsize = 8)
 
    ax[ 1, 1 ].plot(x_train[ fidelities[ :, 0 ] == 0, 0 ], x_train[ fidelities[ :, 0 ] == 0, 1 ], color='blue', marker='x', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 1, 1 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 1, 1 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , pred_std_mf0.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 1, 1 ].set_title("Pred Std Low Fidelity " + name, fontsize = 8)
 
    ax[ 2, 0 ].plot(x_train[ fidelities[ :, 0 ] == 1, 0 ], x_train[ fidelities[ :, 0 ] == 1, 1 ], color='red', marker='o', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 2, 0 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 2, 0 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , pred_mean_mf1.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 2, 0 ].set_title("Pred Mean High Fidelity " + name, fontsize = 8)
 
    ax[ 2, 1 ].plot(x_train[ fidelities[ :, 0 ] == 1, 0 ], x_train[ fidelities[ :, 0 ] == 1, 1 ], color='red', marker='o', markersize=10, linestyle='None')
    if pareto_set is not None:
        ax[ 2, 1 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
    CS = ax[ 2, 1 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , pred_std_mf1.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    ax[ 2, 1 ].set_title("Pred Std High Fidelity " + name, fontsize = 8)

    if pareto_set is not None:
        ax[ 3, 0 ].plot(x_train[ fidelities[ :, 0 ] == 1, 0 ], x_train[ fidelities[ :, 0 ] == 1, 1 ], color='red', marker='o', markersize=10, linestyle='None')
        ax[ 3, 0 ].plot(pareto_set[ : , 0 ], pareto_set[ :, 1 ], color='green', marker='+', markersize=10, linestyle='None')
        CS = ax[ 3, 0 ].contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , y_mean + y_std * sampled_black_box(numpy_grid).reshape((25, 25)))
        plt.clabel(CS, inline=1, fontsize=10)
        ax[ 3, 0 ].set_title("Sample High Fidelity " + name, fontsize = 8)

    plt.savefig(filename, format='pdf', dpi=1000)
    plt.close(fig)

## Mostramos la función de adquisición correspondiente acoplada y desacoplada para cada fidelidad.

def plot_acquisition(numpy_grid, acquisition, blackbox_name, filename, nextpoint = None):

    fig , ax = plt.subplots(1, 1, figsize=(18, 12))
    CS = ax.contour(numpy_grid[ 0 : 25 , 0 ], numpy_grid[ 0 : 25 , 0] , acquisition.reshape((25, 25)))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title("Acquisition " + blackbox_name)
    if nextpoint is not None:
        ax.plot(nextpoint[ 0 ], nextpoint[ 1 ], color='black', marker='x', markersize=10, linestyle='None')
    plt.savefig(filename, format='pdf', dpi=1000)
    plt.close()

def compute_pareto_front(pts): # Corresponds to _cull_algorithm() function of Spearmint

    n_points = pts.shape[ 0 ]

    i_pt = 0
    indices_pareto_pts = np.arange(n_points)

    while i_pt < pts.shape[ 0 ]:
        old_i_pt = indices_pareto_pts[ i_pt ]
        
        # We obtain which points are dominated by the current point

        mask_new_pareto = np.any(pts < pts[ i_pt ], axis=1)
        mask_new_pareto[ i_pt ] = True

        # We remove the dominated points

        indices_pareto_pts = indices_pareto_pts[ mask_new_pareto ]
        pts = pts[ mask_new_pareto ]

        # We update the index taking into account the contraction of 'pts'

        i_pt = np.searchsorted(indices_pareto_pts, old_i_pt, side="right")
            
    mask_pareto_front = np.zeros(n_points, dtype = bool)
    mask_pareto_front[ indices_pareto_pts ] = True

    return mask_pareto_front


# We obtain x and ys for the low fidelity 

x_mf0 = np.random.uniform(size = ((num_inputs_low_fidelity, 2)))
obj1_mf0 = low_fidelity_obj1(x_mf0).reshape((num_inputs_low_fidelity, 1))
obj2_mf0 = low_fidelity_obj2(x_mf0).reshape((num_inputs_low_fidelity, 1))
con1_mf0 = low_fidelity_con1(x_mf0).reshape((num_inputs_low_fidelity, 1))
con2_mf0 = low_fidelity_con2(x_mf0).reshape((num_inputs_low_fidelity, 1))

# We obtain x and ys for the high fidelity 

x_mf1 = np.random.uniform(size = ((num_inputs_high_fidelity, 2)))
obj1_mf1 = high_fidelity_obj1(x_mf1).reshape((num_inputs_high_fidelity, 1))
obj2_mf1 = high_fidelity_obj2(x_mf1).reshape((num_inputs_high_fidelity, 1))
con1_mf1 = high_fidelity_con1(x_mf1).reshape((num_inputs_high_fidelity, 1))
con2_mf1 = high_fidelity_con2(x_mf1).reshape((num_inputs_high_fidelity, 1))

# We iterate through the number of BO iterations

for iteration in range(NUM_BO_ITERS):

    obj1_train_mf0, obj1_train_mf1, obj1_y_mean, obj1_y_std = preprocess_outputs(obj1_mf0, obj1_mf1)
    obj2_train_mf0, obj2_train_mf1, obj2_y_mean, obj2_y_std = preprocess_outputs(obj2_mf0, obj2_mf1)
    con1_train_mf0, con1_train_mf1, con1_y_mean, con1_y_std = preprocess_outputs(con1_mf0, con1_mf1)
    con2_train_mf0, con2_train_mf1, con2_y_mean, con2_y_std = preprocess_outputs(con2_mf0, con2_mf1)

    threshold_constraint_1 = 0.0 * con1_y_std + con1_y_mean
    threshold_constraint_2 = 0.0 * con2_y_std + con2_y_mean

    x_train_mf0 = torch.from_numpy(x_mf0).double()
    x_train_mf1 = torch.from_numpy(x_mf1).double()

    # We obtain x and y for all fidelities

    x_train = torch.cat((x_train_mf1, x_train_mf0), 0)
    obj1_y_train = torch.cat((obj1_train_mf1, obj1_train_mf0), 0)
    obj2_y_train = torch.cat((obj2_train_mf1, obj2_train_mf0), 0)
    con1_y_train = torch.cat((con1_train_mf1, con1_train_mf0), 0)
    con2_y_train = torch.cat((con2_train_mf1, con2_train_mf0), 0)
    fidelities = torch.cat((torch.ones(x_mf1.shape[ 0 ]).double(), torch.zeros(x_mf0.shape[ 0 ]).double()))[ : , None ]

    batch_size = fidelities.shape[ 0 ]

    # We initialize the models taking into account previously_trained_models.

    blackbox_mfdgp_fitter = BlackBoxMFDGPFitter(num_fidelities, batch_size, num_epochs_1=num_epochs_1, num_epochs_2=num_epochs_2)

    previously_trained_model = None

#    if iteration > 0:
#        previously_trained_model = blackbox_mfdgp_fitter_previous_iteration.mfdgp_handlers_objs[ "obj1" ].mfdgp

    blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj1_y_train, fidelities,"obj1" , is_constraint=False, \
           previously_trained_model = previously_trained_model)

#    if iteration > 0:
#        previously_trained_model = blackbox_mfdgp_fitter_previous_iteration.mfdgp_handlers_objs[ "obj2" ].mfdgp

    blackbox_mfdgp_fitter.initialize_mfdgp(x_train, obj2_y_train, fidelities, "obj2", is_constraint=False, \
           previously_trained_model = previously_trained_model)

#    if iteration > 0:
#        previously_trained_model = blackbox_mfdgp_fitter_previous_iteration.mfdgp_handlers_cons[ "con1" ].mfdgp

    blackbox_mfdgp_fitter.initialize_mfdgp(x_train, con1_y_train, fidelities, "con1", \
       threshold_constraint = threshold_constraint_1, is_constraint=True, previously_trained_model = previously_trained_model)

#    if iteration > 0:
#        previously_trained_model = blackbox_mfdgp_fitter_previous_iteration.mfdgp_handlers_cons[ "con2" ].mfdgp

    blackbox_mfdgp_fitter.initialize_mfdgp(x_train, con2_y_train, fidelities, "con2", \
        threshold_constraint = threshold_constraint_2, is_constraint=True, previously_trained_model = previously_trained_model)

    ##########################################################################################################
    # Unconditioned training

    blackbox_mfdgp_fitter.train_mfdgps()

    with open("toy_2D_synthetic/mfdgp_uncond_%dxmf0_%dxmf1_%d_%d_iter_%d.dat"% (num_inputs_low_fidelity, \
        num_inputs_high_fidelity, num_epochs_1, num_epochs_2, iteration), "wb") as fw: \
        pickle.dump(blackbox_mfdgp_fitter, fw)

#    with open("toy_2D_synthetic/mfdgp_uncond_%dxmf0_%dxmf1_%d_%d_iter_%d.dat" % (num_inputs_low_fidelity, \
#        num_inputs_high_fidelity, num_epochs_1, num_epochs_2, iteration), "rb") as fr: \
#        blackbox_mfdgp_fitter = pickle.load(fr)
  
    ##########################################################

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj1" ].mfdgp, low_fidelity_obj1, high_fidelity_obj1, 
        x_train, obj1_y_train, fidelities, obj1_y_mean, obj1_y_std, "Un. obj1", "toy_2D_synthetic/figure_obj1_uncond_iter_" + str(iteration) + ".pdf")

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj2" ].mfdgp, low_fidelity_obj2, high_fidelity_obj2, 
        x_train, obj2_y_train, fidelities, obj2_y_mean, obj2_y_std, "Un. obj2", "toy_2D_synthetic/figure_obj2_uncond_iter_" + str(iteration) + ".pdf")

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con1" ].mfdgp, low_fidelity_con1, high_fidelity_con1,
        x_train, con1_y_train, fidelities, con1_y_mean, con1_y_std, "Un. con1", "toy_2D_synthetic/figure_con1_uncond_iter_" + str(iteration) + ".pdf")

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con2" ].mfdgp, low_fidelity_con2, high_fidelity_con2,
        x_train, con2_y_train, fidelities, con2_y_mean, con2_y_std, "Un. con2", "toy_2D_synthetic/figure_con2_uncond_iter_" + str(iteration) + ".pdf")

    ##########################################################################################################
    # Conditioned training

    pareto_set, pareto_front, sampled_objectives, sampled_cons = blackbox_mfdgp_fitter.sample_and_store_pareto_solution()

    # We store the trained model and copy it for conditional training

    blackbox_mfdgp_fitter_previous_iteration = blackbox_mfdgp_fitter
    blackbox_mfdgp_fitter_cond = deepcopy(blackbox_mfdgp_fitter)

    num_epochs_cond = 1
    blackbox_mfdgp_fitter_cond.num_epochs_1 = 0
    blackbox_mfdgp_fitter_cond.num_epochs_2 = num_epochs_cond

    blackbox_mfdgp_fitter_cond.train_conditioned_mfdgps() 

    with open("toy_2D_synthetic/mfdgp_cond_%dxmf0_%dxmf1_%d_iter_%d.dat"% (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_cond, iteration), "wb") as fw:
        pickle.dump(blackbox_mfdgp_fitter_cond, fw)

    with open("toy_2D_synthetic/mfdgp_sampled_solution_%dxmf0_%dxmf1_%d_iter_%d.dat"% (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_cond, iteration), "wb") as fw:
        pickle.dump((pareto_set, pareto_front, sampled_objectives, sampled_cons), fw)

#    with open("toy_2D_synthetic/mfdgp_cond_%dxmf0_%dxmf1_%d_iter_%d.dat"% (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_cond, iteration), "rb") as fr:
#        blackbox_mfdgp_fitter_cond = pickle.load(fr)

#    with open("toy_2D_synthetic/mfdgp_sampled_solution_%dxmf0_%dxmf1_%d_iter_%d.dat"% (num_inputs_low_fidelity, num_inputs_high_fidelity, num_epochs_cond, iteration), "rb") as fr:
#       pareto_set, pareto_front, sampled_objectives, sampled_cons = pickle.load(fr)

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj1" ].mfdgp, low_fidelity_obj1, high_fidelity_obj1, \
        x_train, obj1_y_train, fidelities, obj1_y_mean, obj1_y_std, "cond obj1", \
        "toy_2D_synthetic/figure_obj1_cond_iter_" + str(iteration) + ".pdf", pareto_set, sampled_objectives[ 0 ])

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj2" ].mfdgp, low_fidelity_obj2, high_fidelity_obj2, \
        x_train, obj2_y_train, fidelities, obj2_y_mean, obj2_y_std, "cond obj2", \
        "toy_2D_synthetic/figure_obj2_cond_iter_" + str(iteration) + ".pdf", pareto_set, sampled_objectives[ 1 ])

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con1" ].mfdgp, low_fidelity_con1, high_fidelity_con1, \
        x_train, con1_y_train, fidelities, con1_y_mean, con1_y_std, "cond con1", \
        "toy_2D_synthetic/figure_con1_cond_iter_" + str(iteration) + ".pdf", pareto_set, sampled_cons[ 0 ])

    plot_black_box(blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con2" ].mfdgp, low_fidelity_con2, high_fidelity_con2,
        x_train, con2_y_train, fidelities, con2_y_mean, con2_y_std, "cond con2", \
        "toy_2D_synthetic/figure_con2_cond_iter_" + str(iteration) + ".pdf", pareto_set, sampled_cons[ 1 ])

    # La distrib_cond debe ser compatible con los datos observados, la frontera y cumplir las cons

    ## Calculamos la adquisición:

    x1 = np.linspace(0, 1, 25).reshape((25, 1))
    x2 = np.linspace(0, 1, 25).reshape((25, 1))
    xx, yy = np.meshgrid(x1, x2)
    numpy_grid = np.vstack((xx.flatten(), yy.flatten())).T
    spacing = torch.from_numpy(numpy_grid)

    jesmoc_mfdgp = JESMOC_MFDGP(model=blackbox_mfdgp_fitter, num_fidelities=num_fidelities, model_cond = blackbox_mfdgp_fitter_cond)
    jesmoc_mfdgp.add_blackbox(0, "obj1", is_constraint=False)
    jesmoc_mfdgp.add_blackbox(0, "obj2", is_constraint=False)
    jesmoc_mfdgp.add_blackbox(0, "con1", is_constraint=True)
    jesmoc_mfdgp.add_blackbox(0, "con2", is_constraint=True)
    jesmoc_mfdgp.add_blackbox(1, "obj1", is_constraint=False)
    jesmoc_mfdgp.add_blackbox(1, "obj2", is_constraint=False)
    jesmoc_mfdgp.add_blackbox(1, "con1", is_constraint=True)
    jesmoc_mfdgp.add_blackbox(1, "con2", is_constraint=True)

    with torch.no_grad():
        acq_obj1_f0 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=0, blackbox_name="obj1", is_constraint=False)
        acq_obj2_f0 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=0, blackbox_name="obj2", is_constraint=False)
        acq_con1_f0 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=0, blackbox_name="con1", is_constraint=True)
        acq_con2_f0 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=0, blackbox_name="con2", is_constraint=True)
        acq_all_f0  = jesmoc_mfdgp.coupled_acq(spacing, fidelity=0)
        acq_obj1_f1 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=1, blackbox_name="obj1", is_constraint=False)
        acq_obj2_f1 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=1, blackbox_name="obj2", is_constraint=False)
        acq_con1_f1 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=1, blackbox_name="con1", is_constraint=True)
        acq_con2_f1 = jesmoc_mfdgp.decoupled_acq(spacing, fidelity=1, blackbox_name="con2", is_constraint=True)
        acq_all_f1  = jesmoc_mfdgp.coupled_acq(spacing, fidelity=1)

    standard_bounds = torch.zeros(2, 2)
    standard_bounds[1] = 1
    
    candidates_0, values_0 = optimize_acqf(acq_function= lambda x: jesmoc_mfdgp.coupled_acq(x, fidelity = 0), bounds=standard_bounds,
        q=1, num_restarts=5, raw_samples=200,  options={"maxiter": 200})

    candidates_1, values_1 = optimize_acqf(acq_function=lambda x: jesmoc_mfdgp.coupled_acq(x, fidelity = 1), bounds=standard_bounds,
        q=1, num_restarts=5, raw_samples=200,  options={"maxiter": 200})

    if values_0 * COST_HIGHER_FIDELITY > values_1 :
        nextpoint = candidates_0[ 0, : ]
        fidelity_to_evaluate = 0
        print("Iter:", iteration, "Acquisition: " + str(values_0.numpy()) + " Evaluating fidelity 0 at", nextpoint.numpy())
    else:
        nextpoint = candidates_1[ 0, : ]
        fidelity_to_evaluate = 1
        print("Iter:", iteration, "Acquisition: " + str(values_1.numpy()) + " Evaluating fidelity 1 at", nextpoint.numpy())

    plot_acquisition(numpy_grid, acq_obj1_f0, 'obj1 f=0', "toy_2D_synthetic/figure_acq_obj1_0_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_obj2_f0, 'obj2 f=0', "toy_2D_synthetic/figure_acq_obj2_0_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_con1_f0, 'con1 f=0', "toy_2D_synthetic/figure_acq_con1_0_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_con2_f0, 'con2 f=0', "toy_2D_synthetic/figure_acq_con2_0_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_all_f0, 'coupled f=0', "toy_2D_synthetic/figure_acq_all_0_iter_" + str(iteration) + ".pdf", nextpoint)
    plot_acquisition(numpy_grid, acq_obj1_f1, 'obj1 f=1', "toy_2D_synthetic/figure_acq_obj1_1_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_obj2_f1, 'obj2 f=1', "toy_2D_synthetic/figure_acq_obj2_1_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_con1_f1, 'con1 f=1', "toy_2D_synthetic/figure_acq_con1_1_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_con2_f1, 'con2 f=1', "toy_2D_synthetic/figure_acq_con2_1_iter_" + str(iteration) + ".pdf")
    plot_acquisition(numpy_grid, acq_all_f1, 'coupled f=1', "toy_2D_synthetic/figure_acq_all_1_iter_" + str(iteration) + ".pdf", nextpoint)

    ## We add the new point to the observed data

    if fidelity_to_evaluate == 0:
        x_mf0 = np.vstack((x_mf0, nextpoint.numpy()))
        obj1_mf0 = np.vstack((obj1_mf0, low_fidelity_obj1(nextpoint.numpy())))
        obj2_mf0 = np.vstack((obj2_mf0, low_fidelity_obj2(nextpoint.numpy())))
        con1_mf0 = np.vstack((con1_mf0, low_fidelity_con1(nextpoint.numpy())))
        con2_mf0 = np.vstack((con2_mf0, low_fidelity_con2(nextpoint.numpy())))
    else:
        x_mf1 = np.vstack((x_mf1, nextpoint.numpy()))
        obj1_mf1 = np.vstack((obj1_mf1, high_fidelity_obj1(nextpoint.numpy())))
        obj2_mf1 = np.vstack((obj2_mf1, high_fidelity_obj2(nextpoint.numpy())))
        con1_mf1 = np.vstack((con1_mf1, high_fidelity_con1(nextpoint.numpy())))
        con2_mf1 = np.vstack((con2_mf1, high_fidelity_con2(nextpoint.numpy())))

    ## We evaluate the candidate solutions in terms of hyper-volume and the optimal hyper-volume.

#    # XXX DHL We resotre the mean and variance:
#
#    data_train = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj1" ].train_dataset.tensors[ 0 ]
#
#    obj1_values = high_fidelity_obj1(data_train.numpy())
#    obj2_values = high_fidelity_obj2(data_train.numpy())
#    con1_values = high_fidelity_con1(data_train.numpy())
#    con2_values = high_fidelity_con2(data_train.numpy())
#
#    obj1_y_mean = np.mean(obj1_values)
#    obj2_y_mean = np.mean(obj2_values)
#    con1_y_mean = np.mean(con1_values)
#    con2_y_mean = np.mean(con2_values)
#
#    obj1_y_std = np.std(obj1_values)
#    obj2_y_std = np.std(obj2_values)
#    con1_y_std = np.std(con1_values)
#    con2_y_std = np.std(con2_values)
#
#    # XXX DHL We resotre the mean and variance:

    from pymoo.indicators.hv import HV
    
    grid = torch.from_numpy(numpy_grid_for_optimal_hv)

    with torch.no_grad():
        with gpytorch.settings.num_likelihood_samples(1):
            pred_means_obj1 = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj1" ].mfdgp.predict_for_acquisition(grid, 1)[ 0 ]
            pred_means_obj2 = blackbox_mfdgp_fitter.mfdgp_handlers_objs[ "obj2" ].mfdgp.predict_for_acquisition(grid, 1)[ 0 ]
            pred_means_con1, pred_vars_con1 = blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con1" ].mfdgp.predict_for_acquisition(grid, 1)
            pred_means_con2, pred_vars_con2 = blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con2" ].mfdgp.predict_for_acquisition(grid, 1)

            pred_vars_con1 = pred_vars_con1 - blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con1" ].mfdgp.hidden_layer_likelihood_1.noise
            pred_vars_con2 = pred_vars_con2 - blackbox_mfdgp_fitter.mfdgp_handlers_cons[ "con2" ].mfdgp.hidden_layer_likelihood_1.noise

    con1_true_values = high_fidelity_con1(numpy_grid_for_optimal_hv)
    con2_true_values = high_fidelity_con2(numpy_grid_for_optimal_hv)
    obj1_true_values = high_fidelity_obj1(numpy_grid_for_optimal_hv)
    obj2_true_values = high_fidelity_obj2(numpy_grid_for_optimal_hv)

    # We undo the standardization

    pred_means_obj1 = pred_means_obj1 * obj1_y_std + obj1_y_mean
    pred_means_obj2 = pred_means_obj2 * obj2_y_std + obj2_y_mean
    pred_means_con1 = pred_means_con1 * con1_y_std + con1_y_mean
    pred_means_con2 = pred_means_con2 * con2_y_std + con2_y_mean
    pred_vars_con1 = pred_vars_con1 * con1_y_std**2
    pred_vars_con2 = pred_vars_con2 * con2_y_std**2

    # We compute highly probable feasible points
    
    dist = torch.distributions.normal.Normal(0, 1)
    to_sel = (dist.cdf(pred_means_con1 / torch.sqrt(pred_vars_con1)) > 0.999) & (dist.cdf(pred_means_con2 / torch.sqrt(pred_vars_con2)) > 0.999)

    feasible_pred_means_obj1 = pred_means_obj1[ to_sel ]
    feasible_pred_means_obj2 = pred_means_obj2[ to_sel ]
    con1_true_values = con1_true_values[ to_sel ]
    con2_true_values = con2_true_values[ to_sel ]
    rec_pareto_set = numpy_grid_for_optimal_hv[ to_sel, : ]

    objectives = np.vstack((feasible_pred_means_obj1.numpy(), feasible_pred_means_obj2.numpy())).T
    index_optimal_points = compute_pareto_front(objectives)
    rec_pareto_set = rec_pareto_set[ index_optimal_points, : ]

    # We remove sub-optimal points. Otherwise it can provide better solutions than the optimal one.

    con1_true_values = con1_true_values[ index_optimal_points ]
    con2_true_values = con2_true_values[ index_optimal_points ]

    feasible = not (np.any(con1_true_values < 0) or np.any(con2_true_values < 0))

    num_optimal_points_ini = rec_pareto_set.shape[ 0 ]
    rec_pareto_set = rec_pareto_set[ con1_true_values >= 0, : ]
    rec_pareto_set = rec_pareto_set[ con2_true_values >= 0, : ]
    num_optimal_points_fini = rec_pareto_set.shape[ 0 ]
    num_infeasible = num_optimal_points_ini - num_optimal_points_fini

    true_objective_values = np.vstack((high_fidelity_obj1(rec_pareto_set), high_fidelity_obj2(rec_pareto_set))).T

    ind =  HV(ref_point=np.array([1000.0, 1000.0]))
    hv_iter = ind(true_objective_values)

    print("Iter:", iteration, "Pareto points found:", rec_pareto_set.shape[ 0 ], "HV recommendation:", hv_iter)

    # Now we compute optimal hypervolume

    obj1_values = high_fidelity_obj1(numpy_grid_for_optimal_hv)
    obj2_values = high_fidelity_obj2(numpy_grid_for_optimal_hv)
    con1_values = high_fidelity_con1(numpy_grid_for_optimal_hv)
    con2_values = high_fidelity_con2(numpy_grid_for_optimal_hv)

    to_sel = (con1_values > 0) & (con2_values > 0)
    feasible_obj1_values = obj1_values[ to_sel ]
    feasible_obj2_values = obj2_values[ to_sel ]

    objectives = np.vstack((feasible_obj1_values, feasible_obj2_values)).T
    index_optimal_points = compute_pareto_front(objectives)
    optimal_points = objectives[ index_optimal_points, : ]
    optimal_hv_iter = ind(optimal_points)

    print("Iter:", iteration, "Optimal optimal_hv:", optimal_hv_iter)

    with open('hypervolumes.txt', 'a') as f:
        print("%lf %lf %lf %lf %lf %lf" % (hv_iter, optimal_hv_iter, float(feasible), \
            num_infeasible, num_optimal_points_fini, num_optimal_points_ini), file = f)

    with open('evaluations.txt', 'a') as f:
        print("%lf" % (fidelity_to_evaluate), file = f)


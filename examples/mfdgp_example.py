import numpy as np
import tqdm
import torch
import gpytorch
import matplotlib.pyplot as plt

# from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader

from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
# from mobocmf.models.ThreeDGPLayer import ThreeDGPLayer # I could make test work with ThreeLayer DGP, also if I the ThreeLayerDGP the traning is much more expensive
from mobocmf.models.TwoDGPLayer import TwoDGPLayer
from mobocmf.mlls.VariationalELBOMF import VariationalELBOMF
from mobocmf.models.DeepGPMultifidelity import DeepGPMultifidelity

NUM_EPOCH = 1000
SPACING   = 1000
N_SAM_LS  = 1000
INI_NOISE = 1e-4

def dgp_train(model, mll, train_loader, num_epochs=100, num_samples_gpytorch=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    # variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1) # This did not work for me woth DeepGP

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        # {'params': model.likelihood.parameters()},
    ], lr=0.01) # Includes GaussianLikelihood parameters

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch", disable=False)
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False, disable=True)
        for (x_batch, y_batch) in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples_gpytorch):
                hyperparameter_optimizer.zero_grad() #; variational_ngd_optimizer.zero_grad()
                                
                output = model(x_batch)
                loss = -mll(output, y_batch.T)
                loss.backward()                
                hyperparameter_optimizer.step() #; variational_ngd_optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())

        # print(i, "loss:", round(loss.item(), 3))

def mfdgp_train(model, mll, train_loader, num_epochs=100, num_samples_gpytorch=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=0.01)
    
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False, disable=True)
        for (x_batch, y_batch, l_fidelities) in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples_gpytorch):                    
                optimizer.zero_grad()
                
                l_output = model.propagate(x_batch)
                loss = -mll(l_output, y_batch.T, l_fidelities=l_fidelities)
                loss.backward()
                optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())

        # print(i, "loss:", round(loss.item(), 3))

def get_mean_and_std_model(inputs, predict_func_model, n_sam_test=25):    
    samples = np.zeros((n_sam_test * 100, 201))
    for i in range(100):
        with gpytorch.settings.num_likelihood_samples(n_sam_test):
            pred_means, pred_variances = predict_func_model(inputs)
            samples[ (i * n_sam_test) : ((i + 1) * n_sam_test), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

    mean = np.mean(samples, 0)
    std  = np.std(samples, 0)

    return mean, std

def plot_mean_and_std_model (ax, x_inputs, mean, std, l_train_x, l_func, l_func_names, low, high, model_name, more_plots=False):

    spacing = np.linspace(low, high, SPACING)[:, None]
    for (func, name_func, train_x) in zip(l_func, l_func_names, l_train_x):
        ax.plot(spacing, func(spacing), "k--", label=name_func)
        
        line, = ax.plot(train_x.numpy(), func(train_x).numpy(), 'r*', markersize=14)
        line.set_label('Observed Data')
        line, = ax.plot(x_inputs.numpy(), mean, 'b-')
        line.set_label('Predictive distribution ' + model_name)
        line = ax.fill_between(x_inputs.numpy()[:,0], (mean + 2.0 * std), (mean - 2.0 * std), alpha=0.5)
        line.set_label('Confidence ' + model_name)
    
    if not more_plots:
        # ax.set_ylim([-3, 3])
        # ax.set_xlim([-1, 1])
        ax.legend()
        plt.show()

def test_mfdgp_and_dgp_models(lower_limit, upper_limit, low_fidelity, high_fidelity, name_low_fidelity, name_high_fidelity,
                              batch_size=20, num_inputs_low_fidelity=40, num_inputs_high_fidelity=10, seed=0):

    assert lower_limit < upper_limit
    assert num_inputs_high_fidelity < num_inputs_low_fidelity

    # We obtain the high fidelity dataset and dataloader
    upper_limit_high_fidelity = (upper_limit - lower_limit) * 0.8 + lower_limit
    x_high = np.random.uniform(lower_limit, upper_limit_high_fidelity, size=(num_inputs_high_fidelity, 1))
    y_high = high_fidelity(x_high)
    x_train_high = torch.from_numpy(x_high).float()
    y_train_high = torch.from_numpy(y_high).float()
    high_fidelity_train_dataset = TensorDataset(x_train_high, y_train_high)
    high_fidelity_train_loader  = DataLoader(high_fidelity_train_dataset, batch_size=batch_size, shuffle=True)

    # We obtain x and y fot the low fidelity 
    x_low = np.random.uniform(lower_limit, upper_limit, size=(num_inputs_low_fidelity - num_inputs_high_fidelity, 1))
    x_low = np.concatenate([x_high, x_low])
    y_low = low_fidelity(x_low)
    x_train_low = torch.from_numpy(x_low).float()
    # y_train_low = torch.from_numpy(y_low).float()
    
    # We obtain the dataset and dataloader for all fidelities
    x   = np.concatenate([x_low, x_high])
    y   = np.concatenate([y_low, y_high])
    fid = torch.cat((torch.ones(len(x_high)).float(),
                     torch.zeros(len(x_low)).float()))[:, None]
    x_train = torch.from_numpy(x).float()
    y_train = torch.from_numpy(y).float()
    all_fidelities_train_dataset = TensorDataset(x_train, y_train, fid)
    all_fidelities_train_loader  = DataLoader(all_fidelities_train_dataset, batch_size=batch_size, shuffle=True)


    # We obtain the value we use to initialice the length-scales
    x_sample = x_train_high[ np.random.choice(np.arange(x_train_high.shape[ 0 ]), size = N_SAM_LS), :  ]
    dist2 = torch.sum(x_sample**2, 1, keepdims = True) - 2.0 * (x_sample).mm(x_sample.T) + torch.sum(x_sample**2, 1, keepdims = True).T
    log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(N_SAM_LS, 1) ]))


    # We create the objects for the model and the approximate marginal log likelihood of the dgp
    dgp_model = TwoDGPLayer(x_train_high.shape, lengthscale=log_l, fixed_noise=INI_NOISE)
    mll = DeepApproximateMLL(VariationalELBO(dgp_model.likelihood, dgp_model, x_train_high.shape[-2]))

    # We train the dgp model using only the high fidelity evals
    dgp_model.train()
    dgp_train(dgp_model, mll, high_fidelity_train_loader, num_epochs=NUM_EPOCH, seed=seed)

    # We evaluate the dgp model in the test inputs
    dgp_model.eval()
    test_inputs = torch.from_numpy(np.linspace(lower_limit, upper_limit, 201)[:,None]).float()
    dgp_mean, dgp_std = get_mean_and_std_model(test_inputs, dgp_model.predict)

    # # We plot the mean and std of the dgp model trained
    # _, ax = plt.subplots(1, 1, figsize=(16, 12))
    # plot_mean_and_std_model(ax, test_inputs, dgp_mean, dgp_std, [x_train_high], [high_fidelity], [name_high_fidelity], lower_limit, upper_limit, model_name="DGP", more_plots=True)


    # We create the objects for the mfdgp model and the approximate marginal log likelihood of the mfdgp
    mfdgp_model = DeepGPMultifidelity(x_train.shape, num_fidelities=2)
    mfdgp_mll = DeepApproximateMLL(VariationalELBOMF(mfdgp_model.likelihood, mfdgp_model, x_train.shape[-2]))

    # We train the mfdgp model using the inputs of all fidelities
    mfdgp_model.train()
    mfdgp_train(mfdgp_model, mfdgp_mll, all_fidelities_train_loader, num_epochs=NUM_EPOCH, num_samples_gpytorch=1, seed=0)

    # We evaluate the dgp model in the test inputs
    mfdgp_model.eval()
    test_inputs = torch.from_numpy(np.linspace(lower_limit, upper_limit, 201)[:,None]).float()
    mfdgp_mean, mfdgp_std = get_mean_and_std_model(test_inputs, mfdgp_model.predict)

    # # We plot the mean and std of the dgp model trained
    # plot_mean_and_std_model(ax, test_inputs, mfdgp_mean, mfdgp_std, [x_train_low, x_train_high], [low_fidelity, high_fidelity],
    #                         [name_low_fidelity, name_high_fidelity], lower_limit, upper_limit, model_name="MFDGP", more_plots=False)


    # We prepare the plots of the models
    spacing = np.linspace(lower_limit, upper_limit, SPACING)[:, None]
    _, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(spacing, low_fidelity(spacing), "b--", label="Low fidelity")
    ax.plot(spacing, high_fidelity(spacing), "r--", label="High fidelity")
    
    line, = ax.plot(x_low, y_low, 'bX', markersize=12)
    line.set_label('Observed Data low fidelity')
    line, = ax.plot(x_high, y_high, 'rX', markersize=12)
    line.set_label('Observed Data high fidelity')

    line, = ax.plot(test_inputs.numpy(), dgp_mean, 'k-')
    line.set_label('Predictive distribution DGP')
    line = ax.fill_between(test_inputs.numpy()[:,0], (dgp_mean + 2.0 * dgp_std), (dgp_mean - 2.0 * dgp_std), color="black", alpha=0.5)
    line.set_label('Confidence DGP')

    line, = ax.plot(test_inputs.numpy(), mfdgp_mean, 'g-')
    line.set_label('Predictive distribution MFDGP')
    line = ax.fill_between(test_inputs.numpy()[:,0], (mfdgp_mean + 2.0 * mfdgp_std), (mfdgp_mean - 2.0 * mfdgp_std), color="green", alpha=0.5)
    line.set_label('Confidence MFDGP')

    # ax.set_ylim([-3, 3])
    # ax.set_xlim([-1, 1])
    ax.legend()
    plt.show()

    import pdb; pdb.set_trace()

if __name__ == "__main__":

    np.random.seed(20)
    torch.manual_seed(20)

    test_mfdgp_and_dgp_models(0.0, 1.0, forrester_mf0, forrester_mf1, "forrester_mf0", "forrester_mf1",
                              batch_size=20, num_inputs_low_fidelity=40, num_inputs_high_fidelity=20)
    test_mfdgp_and_dgp_models(0.0, 1.0, non_linear_sin_mf0, non_linear_sin_mf1, "non_linear_sin_mf0", "non_linear_sin_mf1",
                              batch_size=20, num_inputs_low_fidelity=40, num_inputs_high_fidelity=20)

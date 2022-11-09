import numpy as np
import tqdm
import torch
import gpytorch
import matplotlib.pyplot as plt

# from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader

# from mobocmf.models.ThreeDGPLayer import ThreeDGPLayer # I could make test work with ThreeLayer DGP, also if I the ThreeLayerDGP the traning is much more expensive
from mobocmf.models.TwoDGPLayer import TwoDGPLayer
from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0

# x = np.atleast_2d(np.array([0.18, 0.36, 0.81, 0.94, 0.95, 0.96])).T # Inputs paper mfdgp

def step_function(x):
    return np.sign(x)


HIGH = 1.0
LOW  = 0.0
NUM_INPUTS = 24
NUM_EPOCHS = 1000
NUM_SAMPLES = 1
BATCH_SIZE = int(NUM_INPUTS / 3)
N_SAM_LS = 1000

INI_NOISE = 1e-4

def dgp_train(model, mll, train_x, train_y):

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)    

    model.train()
    # model.likelihood.train()

    # variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1) # This did not work for me woth DeepGP

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        # {'params': model.likelihood.parameters()},
    ], lr=0.01) # Includes GaussianLikelihood parameters

    epochs_iter = tqdm.tqdm(range(NUM_EPOCHS), desc="Epoch", disable=True)

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False, disable=True)
        for x_batch, y_batch in minibatch_iter:
            # print (x_batch,y_batch)
            with gpytorch.settings.num_likelihood_samples(NUM_SAMPLES):
                
                # variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
                                
                output = model(x_batch)
                loss = -mll(output, y_batch.T)
                loss.backward()

                # variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())

        print(i, "loss:", round(loss.item(), 3))

def dgp_test(model, low, high, n_sam_test=25):
    x_test = torch.from_numpy(np.linspace(low, high, 201)[:,None]).float()

    model.eval()
    samples = np.zeros((n_sam_test * 100, 201))
    for i in range(100):
        with gpytorch.settings.num_likelihood_samples(n_sam_test):
            pred_means, pred_variances = model.predict(x_test)
            samples[ (i * n_sam_test) : ((i + 1) * n_sam_test), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

    return samples, x_test

def plot_dgp(samples, train_x, train_y, x_test, spacing_true_x, true_y):
    _, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Get upper and lower confidence bounds
    # Plot training data as black stars

    mean = np.mean(samples, 0)
    std = np.std(samples, 0)
        
    line, = ax.plot(train_x.numpy(), train_y.numpy(), 'r*', markersize=14)
    line.set_label('Observed Data')
    # ax.set_ylim([-3, 3])
    # ax.set_xlim([-1, 1])
    line, = ax.plot(x_test.numpy(), np.mean(samples, 0), 'b-')
    line.set_label('Predictive distribution DGP')
    line = ax.fill_between(x_test.numpy()[:,0], (mean + 2.0 * std), (mean - 2.0 * std), alpha=0.5)
    line.set_label('Confidence DGP')

    ax.plot(spacing_true_x, true_y, "k--", label="true_obj")
    
    ax.legend()
    plt.show()

if __name__ == "__main__":

    np.atleast_2d(np.random.rand(0.0, 1.0, 10))

    np.random.seed(20)
    torch.manual_seed(20)

    # l_low = [-1.0, 0.0, 0.0, 0.0, 0.0]
    # l_high = [1.0]*5
    l_low = [0.0]*4
    l_high = [1.0]*4
    l_funcs = [
        # step_function,
        forrester_mf1,
        forrester_mf0,
        non_linear_sin_mf1,
        non_linear_sin_mf0]

    x = np.atleast_2d(np.random.rand(NUM_INPUTS)).T

    for low, high, func in zip(l_low, l_high, l_funcs):
        y = func(x)

        # plt.plot(x, y, 'o', color='black')

        train_x = torch.from_numpy(x).float()
        train_y = torch.from_numpy(y).float()

        x_sample = train_x[ np.random.choice(np.arange(train_x.shape[ 0 ]), size = N_SAM_LS), :  ]
        dist2 = torch.sum(x_sample**2, 1, keepdims = True) - 2.0 * (x_sample).mm(x_sample.T) + torch.sum(x_sample**2, 1, keepdims = True).T
        log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(N_SAM_LS, 1) ]))

        model = TwoDGPLayer(train_x.shape, lengthscale=log_l, fixed_noise=INI_NOISE) #, likelihood=likelihood)

        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

        dgp_train(model, mll, train_x, train_y)

        samples, x_test = dgp_test(model, low, high)

        spacing_true_x = np.linspace(low, high, 1000)[:, None]
        true_y = func(spacing_true_x)
        plot_dgp(samples, train_x, train_y, x_test, spacing_true_x, true_y)

        import pdb;pdb.set_trace()


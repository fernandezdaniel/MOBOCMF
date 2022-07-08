from tabnanny import verbose
import numpy as np
import torch
import tqdm
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ExactGP
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.mlls import DeepApproximateMLL

from mlls.VariationalELBOMultifidelity import VariationalELBOMultifidelity
from models.DeepGPMultifidelity import DeepGPMultifidelity

from gpytorch.mlls import VariationalELBO
from models.MyDeepGP import MyDeepGP

plt_state_iters = False

LOW  = 0.0
HIGH = 1.0
# LOC_MAXIMUM = 0.5

NUM_FIDELITIES = 2

def obj_mf1(x, sd=0):
    """
    High fidelity version of nonlinear sin function
    """

    return (x - np.sqrt(2)) * obj_mf0(x, 0) ** 2 + np.random.randn(x.shape[0], 1) * sd

def obj_mf0(x, sd=0):
    """
    Low fidelity version of nonlinear sin function
    """

    return np.sin(8 * np.pi * x) + np.random.randn(x.shape[0], 1) * sd

# def obj_mf1(X):
#     return (3-X)*np.sin(X*10) + X*1.2

# def obj_mf0(X):
#     return 1.5*(3-X)*np.sin(X*10) + X*1.2 + 1.44

def train_dgpmf(train_x, train_loader, num_epochs=300, num_samples=1, num_hidden_layers=2, plt_state_iters=False, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    model = DeepGPMultifidelity(train_x.shape, num_hidden_layers=num_hidden_layers)
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=0.01)
    
    mll = DeepApproximateMLL(VariationalELBOMultifidelity(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

    if plt_state_iters:
        l_loss = np.zeros(num_epochs)
        plt.ion()
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        # n_iter = 0
        for (x_batch, y_batch, l_fidelities) in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):                    
                optimizer.zero_grad()
                
                l_output = model.propagate(x_batch)
                loss = -mll(l_output, y_batch.T, l_fidelities=l_fidelities)
                loss.backward()
                optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())
                

                if plt_state_iters:
                    global fig, ax1, ax2

                    l_loss[i] = float(str(loss).replace("tensor(", "").split(", grad_fn")[0])

                    # hl2_00_ls = list(list(model.hidden_layer_2.covar_module.kernels._modules.items())[0][1].kernels)[0].base_kernel.raw_lengthscale[0,0]
                    # hl2_00_ls = str(hl2_00_ls).split("(")[1].split(", grad")[0]

                    # hl2_01_ls = list(list(model.hidden_layer_2.covar_module.kernels._modules.items())[0][1].kernels[1].kernels._modules.items())[0][1].base_kernel.raw_lengthscale[0,0]
                    # hl2_01_ls = str(hl2_01_ls).split("(")[1].split(", grad")[0]
                    # # hl2_11_ls = list(list(model.hidden_layer_2.covar_module.kernels._modules.items())[0][1].kernels[1].kernels._modules.items())[1][1].base_kernel.raw_lengthscale[0,0]
                    # # hl2_11_ls = str(hl2_11_ls).split("(")[1].split(", grad")[0]

                    

                    fig.suptitle("Num iter: " + str(i) + "/" + str(num_epochs) + ", loss: "+ str(l_loss[i])) # + ", hl2_00_ls: " + str(hl2_00_ls) + ", hl2_01_ls: " + str(hl2_01_ls))
                    ax1.set_ylim(-5, 5)
                    ax1.plot(x_batch[:,0], y_batch[:,0], "ro")
                    ax1.plot(x_batch[:,0], y_batch[:,0], "bo")
                    ax1.plot(x_batch[l_fidelities == 0], y_batch[l_fidelities == 0], 'bo', label="Fidelity 0")
                    ax1.plot(x_batch[l_fidelities == 1], y_batch[l_fidelities == 1], 'ro', label="Fidelity 1")
                    ax1.plot(torch.linspace(LOW, HIGH, 100), model.predict(torch.linspace(LOW, HIGH, 100).reshape(shape=(100,1)))[0][0], "*")
                    ax2.plot(np.arange(num_epochs), l_loss, "ko")
                    ax1.legend(loc="upper left")
                    plt.draw()
                    plt.pause(0.0001)
                    ax1.clear()
                    ax2.clear()

    return str(loss).split("r(")[1].split(", ")[0], model

def train_dgp(train_x, train_loader, num_epochs=300, num_samples=1, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    model = MyDeepGP(train_x.shape)
    optimizer = torch.optim.Adam([ {'params': model.parameters()}, ], lr=0.05)

    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

    for _ in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                
                optimizer.zero_grad()
                                
                output = model(x_batch)
                loss = -mll(output, y_batch.T)
                loss.backward()
                optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())

    return str(loss).split("r(")[1].split(", ")[0], model

def compute_mean_and_std_model(model, x_test, num_samples_test=40, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    model.eval()
    dgpmf_samples = np.zeros((num_samples_test * 100, 201))
    for i in range(100):
        with gpytorch.settings.num_likelihood_samples(num_samples_test):
            predictive_means, predictive_variances = model.predict(x_test)
            dgpmf_samples[ (i * num_samples_test) : ((i + 1) * num_samples_test), : ] = np.random.normal(size = predictive_means.numpy().shape) * \
                 np.sqrt(predictive_variances.numpy()) + predictive_means.numpy()

    mean = np.mean(dgpmf_samples, 0)
    std  = np.std(dgpmf_samples, 0)

    return mean, std

def test_mf_deep_gp_example(num_fidelities=NUM_FIDELITIES, plt_state_iters=False, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)

    num_inputs = 70
    x_train = np.random.uniform(low=LOW, high=HIGH, size=(num_inputs, 1))
    y_train = obj_mf0(x_train)
    l_input_fidelities = np.random.randint(num_fidelities, size=(num_inputs, 1))
    l_input_fidelities[x_train >= (LOW + (HIGH - LOW)*0.65)] = 0
    y_train[l_input_fidelities == 1] = obj_mf1(x_train[l_input_fidelities == 1][:,None])[...,0]
    

    # x_train_l = np.linspace(0, 1, 50)[:, None]
    # x_train_h = x_train_l[::4, :][:-3]
    # y_train_l = obj_mf0(x_train_l)
    # y_train_h = obj_mf1(x_train_h)
    # x_train = np.concatenate([x_train_l, x_train_h])
    # y_train = np.concatenate([y_train_l, y_train_h])
    # l_input_fidelities = torch.from_numpy(np.concatenate([np.zeros((len(x_train_l), 1)),
    #                                                       np.ones((len(x_train_h), 1))])).float()

    # plt.plot(x_train[l_input_fidelities == 0], y_train[l_input_fidelities == 0], 'bo')
    # plt.plot(x_train[l_input_fidelities == 1], y_train[l_input_fidelities == 1], 'ro')
    # plt.show()

    l_input_fidelities = torch.from_numpy(l_input_fidelities)
    dgpmf_train_x = torch.from_numpy(x_train).float()
    dgpmf_train_y = torch.from_numpy(y_train).float()
    dgp_train_x   = dgpmf_train_x[l_input_fidelities == (num_fidelities-1)][:,None] + 0.0 
    dgp_train_y   = dgpmf_train_y[l_input_fidelities == (num_fidelities-1)][:,None] + 0.0 

    dgpmf_train_dataset = TensorDataset(dgpmf_train_x, dgpmf_train_y, l_input_fidelities)
    dgpmf_train_loader  = DataLoader(dgpmf_train_dataset, batch_size=70, shuffle=True)
    dgp_train_dataset   = TensorDataset(dgp_train_x, dgp_train_y)
    dgp_train_loader    = DataLoader(dgp_train_dataset, batch_size=70, shuffle=True)

    x_test = torch.from_numpy(np.linspace(LOW, HIGH, 201)[:,None]).float()

    dgpmf_loss, DGPMF_model = train_dgpmf(dgpmf_train_x, dgpmf_train_loader, num_hidden_layers=num_fidelities, plt_state_iters=plt_state_iters, seed=seed)
    dgpmf_mean, dgpmf_std = compute_mean_and_std_model(DGPMF_model, x_test, seed=seed)
    
    dgp_loss, DGP_model = train_dgp(dgp_train_x, dgp_train_loader, seed=seed)
    dgp_mean, dgp_std = compute_mean_and_std_model(DGP_model, x_test, seed=seed)

    dgp0_train_x   = dgpmf_train_x[l_input_fidelities == 0][:,None] + 0.0 
    dgp0_train_y   = dgpmf_train_y[l_input_fidelities == 0][:,None] + 0.0 
    dgp0_train_dataset = TensorDataset(dgp0_train_x, dgp0_train_y)
    dgp0_train_loader  = DataLoader(dgp0_train_dataset, batch_size=10, shuffle=True)
    dgp0_loss, DGP0_model = train_dgp(dgp0_train_x, dgp0_train_loader, seed=seed)
    dgp0_mean, dgp0_std = compute_mean_and_std_model(DGP0_model, x_test, seed=seed)

    # We plot the results

    _, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot(x_train[l_input_fidelities == 0], y_train[l_input_fidelities == 0], 'bo', label="Fidelity 0")
    ax.plot(x_train[l_input_fidelities == 1], y_train[l_input_fidelities == 1], 'ro', label="Fidelity 1")
    ax.plot(np.linspace(LOW, HIGH, 1000), obj_mf1(np.linspace(LOW, HIGH, 1000)[:,None])[...,0], 'b-')
    ax.plot(np.linspace(LOW, HIGH, 1000), obj_mf1(np.linspace(LOW, HIGH, 1000)[:,None])[...,0], 'r-', label="Ground truth")
    ax.set_xlim([LOW, HIGH])

    line, = ax.plot(x_test.numpy(), dgp_mean, 'k--')
    line.set_label('Predictive distribution DGP1 (loss: ' + dgp_loss + ')')
    line = ax.fill_between(x_test.numpy()[:,0], (dgp_mean + 2.0 * dgp_std), (dgp_mean - 2.0 * dgp_std), alpha=0.1, color="k")
    line.set_label('Confidence DGP')

    line, = ax.plot(x_test.numpy(), dgp0_mean, 'm--')
    line.set_label('Predictive distribution DGP0 (loss: ' + dgp0_loss + ')')
    line = ax.fill_between(x_test.numpy()[:,0], (dgp0_mean + 2.0 * dgp0_std), (dgp0_mean - 2.0 * dgp0_std), alpha=0.1, color="m")
    line.set_label('Confidence DGP')

    line, = ax.plot(x_test.numpy(), dgpmf_mean, 'g--')
    line.set_label('Predictive distribution DGPMF (loss: ' + dgpmf_loss + ')')
    line = ax.fill_between(x_test.numpy()[:,0], (dgpmf_mean + 2.0 * dgpmf_std), (dgpmf_mean - 2.0 * dgpmf_std), alpha=0.1, color="g")
    line.set_label('Confidence DGPMF')

    ax.legend()

    plt.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":

    plt_state_iters = False

    if plt_state_iters:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    test_mf_deep_gp_example(plt_state_iters=plt_state_iters)

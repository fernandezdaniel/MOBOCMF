import numpy as np
import tqdm
import torch
import gpytorch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from mobocmf.models.ExactGPModel import ExactGPModel
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0 
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.models.DeepGPMultifidelity import DeepGPMultifidelity
from mobocmf.mlls.VariationalELBOMF import VariationalELBOMF

# We obtain the high fidelity dataset and dataloader

np.random.seed(0)

num_inputs_high_fidelity = 10
num_inputs_low_fidelity = 50

num_epochs_1 = 5000
num_epochs_2 = 15000
batch_size = num_inputs_low_fidelity + num_inputs_high_fidelity

upper_limit = 1.0
lower_limit = 0.0

high_fidelity = non_linear_sin_mf1
low_fidelity = non_linear_sin_mf0

# We obtain x and y for the low fidelity 

#x_low = np.random.uniform(lower_limit, upper_limit, size=(num_inputs_low_fidelity, 1))
x_low = np.linspace(0, 1.0, num_inputs_low_fidelity).reshape((num_inputs_low_fidelity, 1))
y_low = low_fidelity(x_low)

# We obtain x and y for the high fidelity 

upper_limit_high_fidelity = (upper_limit - lower_limit) * 0.7 + lower_limit
#x_high = np.random.uniform(lower_limit, upper_limit_high_fidelity, size=(num_inputs_high_fidelity, 1))
#x_high = np.linspace(0, 0.7, num_inputs_high_fidelity).reshape((num_inputs_high_fidelity, 1))
x_high = np.array([0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]).reshape((num_inputs_high_fidelity, 1))
y_high = high_fidelity(x_high)

# Important: DHL we use the same mean and standar devaition for each fidelity !!!

y_high_mean = np.mean(np.vstack((y_high, y_low)))
y_high_std = np.std(np.vstack((y_high, y_low)))
y_low_mean = y_high_mean 
y_low_std = y_high_std 

y_high = (y_high - y_high_mean) / y_high_std
x_train_high = torch.from_numpy(x_high).double()
y_train_high = torch.from_numpy(y_high).double()

y_low = (y_low - y_low_mean) / y_low_std
x_train_low = torch.from_numpy(x_low).double()
y_train_low = torch.from_numpy(y_low).double()
    
# We obtain x and y for all fidelities

x = np.concatenate([x_high, x_low])
y = np.concatenate([y_high, y_low])
fid = torch.cat((torch.ones(len(x_high)).double(), torch.zeros(len(x_low)).double()))[:, None] 

x_train = torch.from_numpy(x).double()
y_train = torch.from_numpy(y).double()
all_fidelities_train_dataset = TensorDataset(x_train, y_train, fid)
all_fidelities_train_loader  = DataLoader(all_fidelities_train_dataset, batch_size=batch_size, shuffle=True)

# We create the objects for the mfdgp model and the approximate marginal log likelihood of the mfdgp

model = DeepGPMultifidelity(x_train, y_train, fid, num_fidelities = 2)
model.double()
elbo = VariationalELBOMF(model, x_train.shape[-2], 2)

model.fix_variational_hypers(True)

optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=0.003) 

for i in range(num_epochs_1):

    minibatch_iter = all_fidelities_train_loader
    loss_iter = 0.0

    for (x_batch, y_batch, fidelities) in minibatch_iter:

        with gpytorch.settings.num_likelihood_samples(1):
            optimizer.zero_grad() 
            output = model(x_batch)
            loss = -elbo(output, y_batch.T, fidelities)
            loss.backward()                
            optimizer.step() 
            loss_iter += loss

    print("Epoch:", i, "/", num_epochs_1, ". Avg. Neg. ELBO per epoch:", loss_iter.item())

model.fix_variational_hypers(False)

optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=0.001) 

for i in range(num_epochs_2):

    minibatch_iter = all_fidelities_train_loader
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
#
# We compute mean and std of the predictive distribution via Monte Carlo

model.eval()

test_inputs = torch.from_numpy(np.linspace(lower_limit, upper_limit, 200)[:,None]).double()
samples = np.zeros((100, test_inputs.shape[ 0 ]))

for i in range(100):
    with gpytorch.settings.num_likelihood_samples(1):
        pred_means, pred_variances = model.predict(test_inputs, 1)
        samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

pred_mean_high = np.mean(samples, 0) * y_high_std + y_high_mean
pred_std_high  = np.std(samples, 0) * y_high_std

for i in range(100):
    with gpytorch.settings.num_likelihood_samples(1):
        pred_means, pred_variances = model.predict(test_inputs, 0)
        samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

pred_mean_low = np.mean(samples, 0) * y_low_std + y_low_mean
pred_std_low = np.std(samples, 0) * y_low_std

# We fit a standard GP Model

training_iter = 1000
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 1e-3
gp_model = ExactGPModel(x_train_high, y_train_high.T, likelihood)
gp_model.train()
likelihood.train()

# Use the adam optimizer

optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = gp_model(x_train_high)
    loss = -mll(output, y_train_high.T)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(), gp_model.covar_module.base_kernel.lengthscale.item(), gp_model.likelihood.noise.item()))
    optimizer.step()

gp_model.eval()
likelihood.eval()

with torch.no_grad():
    y_preds = likelihood(gp_model(test_inputs))

y_mean_gp = y_preds.mean.numpy() * y_high_std + y_high_mean
y_var_gp = y_preds.variance.numpy() * y_high_std**2

spacing = np.linspace(lower_limit, upper_limit, 1000)[:, None]
_, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.plot(spacing, low_fidelity(spacing), "b--", label="Low fidelity")
ax.plot(spacing, high_fidelity(spacing), "r--", label="High fidelity")

line, = ax.plot(x_low, y_low * y_low_std + y_low_mean, 'bX', markersize=12)
line.set_label('Observed Data low fidelity')
line, = ax.plot(x_high, y_high * y_high_std + y_high_mean, 'rX', markersize=12)
line.set_label('Observed Data high fidelity')

line, = ax.plot(test_inputs.numpy(), pred_mean_high, 'g-')
line.set_label('Predictive distribution MFDGP High Fidelity')
line = ax.fill_between(test_inputs.numpy()[:,0], (pred_mean_high + pred_std_high), \
    (pred_mean_high - pred_std_high), color="green", alpha=0.5)
line.set_label('Confidence MFDGP High Fidelity')

line, = ax.plot(test_inputs.numpy(), pred_mean_low, 'r-')
line.set_label('Predictive distribution MFDGP Low Fidelity')
line = ax.fill_between(test_inputs.numpy()[:,0], (pred_mean_low + pred_std_low), \
    (pred_mean_low- pred_std_low), color="red", alpha=0.5)
line.set_label('Confidence MFDGP Low Fidelity')

line, = ax.plot(test_inputs.numpy(), y_mean_gp, 'b-')
line.set_label('Predictive distribution GP High Fidelity')
line = ax.fill_between(test_inputs.numpy()[:,0], (y_mean_gp + np.sqrt(y_var_gp)), \
    (y_mean_gp - np.sqrt(y_var_gp)), color="blue", alpha=0.5)
line.set_label('Confidence GP High Fidelity')

ax.legend()
plt.show()

import pdb; pdb.set_trace()

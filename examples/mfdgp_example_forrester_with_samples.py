import numpy as np
import tqdm
import torch
import gpytorch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from mobocmf.models.ExactGPModel import ExactGPModel
from mobocmf.test_functions.forrester import forrester_mf1, forrester_mf0
from mobocmf.test_functions.non_linear_sin import non_linear_sin_mf1, non_linear_sin_mf0
from mobocmf.models.DeepGPMultifidelity import DeepGPMultifidelity
from mobocmf.mlls.VariationalELBOMF import VariationalELBOMF

# We obtain the high fidelity dataset and dataloader

np.random.seed(0)

num_inputs_high_fidelity = 4
num_inputs_low_fidelity = 12

num_epochs_1 = 600
num_epochs_2 = 900
batch_size = num_inputs_low_fidelity + num_inputs_high_fidelity

upper_limit = 1.0
lower_limit = 0.0

high_fidelity = forrester_mf1
low_fidelity = forrester_mf0

# We obtain x and y for the low fidelity 

#x_low = np.random.uniform(lower_limit, upper_limit, size=(num_inputs_low_fidelity, 1))
x_low = np.linspace(0, 1.0, num_inputs_low_fidelity).reshape((num_inputs_low_fidelity, 1))
y_low = low_fidelity(x_low)

# We obtain x and y for the high fidelity 

upper_limit_high_fidelity = (upper_limit - lower_limit) * 0.7 + lower_limit
#x_high = np.random.uniform(lower_limit, upper_limit_high_fidelity, size=(num_inputs_high_fidelity, 1))
#x_high = np.linspace(0, 0.7, num_inputs_high_fidelity).reshape((num_inputs_high_fidelity, 1))
x_high = np.array([0.1, 0.3, 0.5, 0.7]).reshape((num_inputs_high_fidelity, 1))
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
model.double() # We use double precission to avoid numerical problems
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

model.eval()

#############################################################################################
# RFF sampling
SPACING_RFF = 200
nFeatures = 2000

test_inputs = torch.from_numpy(np.linspace(lower_limit, upper_limit, SPACING_RFF)[:,None]).double()
x_t = test_inputs.numpy()

N_SAMPLES = 10
samples_l0 = np.zeros((N_SAMPLES, test_inputs.shape[ 0 ]))
samples_l1 = np.zeros((N_SAMPLES, test_inputs.shape[ 0 ]))
for i in range(N_SAMPLES):
    sample_l0 = model.hidden_layer_0.sample_from_posterior(x_t.shape[-1],
                                                        likelihood=model.hidden_layer_likelihood_0,
                                                        nFeatures=nFeatures,
                                                        seed=0)


    sample_l1 = model.hidden_layer_1.sample_from_posterior(x_t.shape[-1],
                                                        likelihood=model.hidden_layer_likelihood_1,
                                                        sample_from_posterior_last_layer=sample_l0,
                                                        nFeatures=nFeatures,
                                                        seed=i)

    samples_l0[ i : (i + 1), : ] = sample_l0(x_t)
    samples_l1[ i : (i + 1), : ] = sample_l1(x_t)

#############################################################################################


test_inputs = torch.from_numpy(np.linspace(lower_limit, upper_limit, 200)[:,None]).double()
samples = np.zeros((250, test_inputs.shape[ 0 ]))

with gpytorch.settings.num_likelihood_samples(1): 
    ret = model.hidden_layer_1(torch.from_numpy(np.hstack((test_inputs[:,0:1], test_inputs[:,0:1]))))

for i in range(250):
    with gpytorch.settings.num_likelihood_samples(1):
        pred_means, pred_variances = model.predict(test_inputs, 1)
        samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

pred_mean_high = np.mean(samples, 0) * y_high_std + y_high_mean
pred_std_high  = np.std(samples, 0) * y_high_std

for i in range(250):
    with gpytorch.settings.num_likelihood_samples(1):
        pred_means, pred_variances = model.predict(test_inputs, 0)
        samples[ i : (i + 1), : ] = np.random.normal(size = pred_means.numpy().shape) * \
                np.sqrt(pred_variances.numpy()) + pred_means.numpy()

pred_mean_low = np.mean(samples, 0) * y_low_std + y_low_mean
pred_std_low = np.std(samples, 0) * y_low_std

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


for sample in samples_l0: ax.plot(test_inputs.numpy(), sample * y_high_std + y_high_mean)
for sample in samples_l1: ax.plot(test_inputs.numpy(), sample * y_high_std + y_high_mean)

 
ax.legend()
plt.show()

import pdb; pdb.set_trace()



# Pruebas para comprobar que el calculo del kernel es correcto

# import scipy.linalg as spla

# model0 = model.hidden_layer_0
# input_dim=1
# likelihood=model.hidden_layer_likelihood_0
# nFeatures=100000
# seed=0

# x_data = model0.variational_strategy.inducing_points.detach().numpy()
# y_data = model0.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]
# S = model0.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()
# print("S:\n", S[1:5,1:5])

# print("\ndiff indu - x_data:\n", np.sum((model.hidden_layer_0.variational_strategy.inducing_points - x_data).detach().numpy()))

# lengthscale = model0.covar_module.base_kernel.lengthscale.detach().numpy().item()
# raw_lengthscale = model0.covar_module.base_kernel.raw_lengthscale.detach().numpy().item()
# alpha = model0.covar_module.outputscale.detach().numpy().item()
# raw_alpha = model0.covar_module.raw_outputscale.detach().numpy().item()
# sigma2 =1e-8; print("\nsigma2:", likelihood.noise.detach().numpy().item())

# print("lengthscale:", lengthscale)
# print("raw_lengthscale:", raw_lengthscale)
# print("alpha:", alpha)
# print("raw_alpha:", raw_alpha)
# print("sigma2:", sigma2)

# def phi_rbf(x, W, b, alpha, nFeatures):
#     return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)

# def chol2inv(chol):
#     return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

# W  = np.random.normal(size=(nFeatures, input_dim)) / lengthscale
# b  = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))
# randomness  = np.random.normal(loc=0., scale=1., size=nFeatures)

# Phi = phi_rbf(x_data, W, b, alpha, nFeatures)

# sol = Phi.T @ Phi
# print("\nPhi @ Phi.T:\n", sol[1:5,1:5])

# kernel_x_data = model.hidden_layer_0.covar_module.base_kernel(model.hidden_layer_0.variational_strategy.inducing_points)
# # print("\nkernel_x_data.base_kernel:\n", kernel_x_data[1:5,1:5].detach().numpy())

# kernel_x_data = model.hidden_layer_0.covar_module(model.hidden_layer_0.variational_strategy.inducing_points)
# print("\nkernel_x_data:\n", kernel_x_data[1:5,1:5].detach().numpy())

# from sklearn.metrics.pairwise import rbf_kernel


# # print("\nx_data[1:5]:\n", x_data[1:5])

# import numexpr as ne
# from scipy.linalg.blas import sgemm

# def k_rbf(X, lengthscale, var):
#     dist_X = (X[:, None] - X[None, :])[:,:,0]
#     return ne.evaluate('v * exp(- (D ** 2) / (2 * l ** 2))', {
#         'D' : dist_X,
#         'l' : lengthscale,
#         'v' : var
#     })

# print("\nK:\n", k_rbf(x_data[1:5], lengthscale=lengthscale, var=alpha))







# import scipy.linalg as spla

# def rff_sample_posterior_weights(x_data, y_data, S, W, b, nFeatures=200, alpha=1.0, sigma2=1e-6):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)
    
#     def chol2inv(chol):
#         return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

#     randomness  = np.random.normal(loc=0., scale=1., size=nFeatures)

#     Phi = phi_rbf(x_data, W, b, alpha, nFeatures)
    
#     A = Phi @ Phi.T + sigma2*np.eye(nFeatures)
#     chol_A_inv = spla.cholesky(A)
#     A_inv = chol2inv(chol_A_inv)

#     m = spla.cho_solve((chol_A_inv, False), Phi @ y_data)
#     extraVar = 0.0 # A_inv @ Phi @ S @ Phi.T @ A_inv
#     return m + (randomness @ spla.cholesky(sigma2*A_inv + extraVar, lower=False)).T

# def sample_from_posterior_layer0(model, input_dim, likelihood, nFeatures=200, seed=None):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)

#     np.random.seed(seed)

#     x_data = model.variational_strategy.inducing_points.detach().numpy(); print(x_data.ravel())
#     y_data = model.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]; print(y_data.ravel())
#     S_data = model.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()

#     lengthscale =1.0; print("lengthscale:", model.covar_module.base_kernel.raw_lengthscale.detach().numpy().item())
#     alpha =1.0; print("alpha:", model.covar_module.raw_outputscale.detach().numpy().item())
#     sigma2 =1e-6; print("sigma2:", likelihood.noise.detach().numpy().item())

#     W  = np.random.normal(size=(nFeatures, input_dim)) / lengthscale
#     b  = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

#     theta = rff_sample_posterior_weights(x_data, y_data[:, 0], S_data, W, b, nFeatures=nFeatures, alpha=alpha, sigma2=sigma2)
    
#     def wrapper(x):
#         features = phi_rbf(x, W, b, alpha, nFeatures)
#         return theta @ features

#     return wrapper

# sa = sample_from_posterior_layer0(model.hidden_layer_0, x_t.shape[-1], likelihood=model.hidden_layer_likelihood_0, nFeatures=nFeatures, seed=0)

# plt.plot(np.array([0., 0.09090909, 0.18181818, 0.27272727, 0.36363636,
#                   0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
#                   0.90909091, 1.]),
#         np.array([-0.47065825, -0.63522399, -0.48852088, -0.24559148, -0.0570804 ,
#                    0.16645165,  0.37990257,  0.31763058,  0.10252796,  0.45339102,
#                    1.67323949,  2.70487143]) * y_high_std + y_high_mean, "Xk")

# plt.plot(x_t, sa(x_t) * y_high_std + y_high_mean)







# import scipy.linalg as spla

# def rff_sample_posterior_weights(x_data, y_data, S, W, b, nFeatures=200, alpha=1.0, sigma2=1e-6):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)
    
#     def chol2inv(chol):
#         return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

#     randomness  = np.random.normal(loc=0., scale=1., size=nFeatures)

#     Phi = phi_rbf(x_data, W, b, alpha, nFeatures)
    
#     A = Phi @ Phi.T + sigma2*np.eye(nFeatures)
#     chol_A_inv = spla.cholesky(A)
#     A_inv = chol2inv(chol_A_inv)

#     m = spla.cho_solve((chol_A_inv, False), Phi @ y_data)
#     extraVar = 0.0 # A_inv @ Phi @ S @ Phi.T @ A_inv
#     return m + (randomness @ spla.cholesky(sigma2*A_inv + extraVar, lower=False)).T

# def sample_from_posterior_layer0(model, input_dim, likelihood, nFeatures=200, seed=None):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W @ x.T + b)

#     np.random.seed(seed)

#     x_data = model.variational_strategy.inducing_points.detach().numpy(); print(x_data.ravel())
#     y_data = model.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]; print(y_data.ravel())
#     S_data = model.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()

#     lengthscale = model.covar_module.base_kernel.raw_lengthscale.detach().numpy().item()
#     alpha = model.covar_module.raw_outputscale.detach().numpy().item()
#     sigma2 =1e-6; print("sigma2:", likelihood.noise.detach().numpy().item())

#     W  = np.random.normal(size=(nFeatures, input_dim)) / lengthscale
#     b  = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

#     theta = rff_sample_posterior_weights(x_data, y_data[:, 0], S_data, W, b, nFeatures=nFeatures, alpha=alpha, sigma2=sigma2)
    
#     def wrapper(x):
#         features = phi_rbf(x, W, b, alpha, nFeatures)
#         return theta @ features

#     return wrapper

# sa = sample_from_posterior_layer0(model.hidden_layer_0, x_t.shape[-1], likelihood=model.hidden_layer_likelihood_0, nFeatures=nFeatures, seed=0)

# plt.plot(np.array([0., 0.09090909, 0.18181818, 0.27272727, 0.36363636,
#                   0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
#                   0.90909091, 1.]),
#         np.array([-0.47065825, -0.63522399, -0.48852088, -0.24559148, -0.0570804 ,
#                    0.16645165,  0.37990257,  0.31763058,  0.10252796,  0.45339102,
#                    1.67323949,  2.70487143]) * y_high_std + y_high_mean, "Xk")

# plt.plot(x_t, sa(x_t) * y_high_std + y_high_mean)







# import scipy.linalg as spla

# def rff_sample_posterior_weights(x_data, y_data, S, W, b, nFeatures=2, alpha=1.0, sigma2=1e-6):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W.dot(x.T) + b)
    
#     def chol2inv(chol):
#         return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

#     randomness  = np.random.normal(loc=0., scale=1., size=nFeatures)

#     Phi = phi_rbf(x_data, W, b, alpha, nFeatures)

#     A = Phi.dot(Phi.T) + sigma2*np.eye(nFeatures)
#     chol_A_inv = spla.cholesky(A)
#     A_inv = chol2inv(chol_A_inv)

#     m = spla.cho_solve((chol_A_inv, False), Phi.dot(y_data))
#     theta = m + (randomness.dot(spla.cholesky(sigma2*A_inv, lower=False))).T

#     extraVar = 0.0 # A_inv @ Phi @ S @ Phi.T @ A_inv
#     return m + (randomness.dot(spla.cholesky(sigma2*A_inv + extraVar, lower=False))).T

# def sample_from_posterior_layer0(model, input_dim, likelihood, nFeatures=200, seed=None):
#     def phi_rbf(x, W, b, alpha, nFeatures):
#         return np.sqrt(2.0 * alpha / nFeatures) * np.cos(W.dot(x.T) + b)

#     np.random.seed(seed)

#     x_data = np.linspace(-8., 8., 30)[:, None] #np.random.uniform(LOW, HIGH, num_data)[:, None]
#     y_data = np.sin(x_data) # Noiseless observations #+ 1e-1 * np.random.normal(size=x_data.shape)

#     #x_data = model.variational_strategy.inducing_points.detach().numpy(); print(x_data.ravel())
#     #y_data = model.variational_strategy.variational_distribution.mean.detach().numpy()[:, None]; print(y_data.ravel())
#     S_data = model.variational_strategy.variational_distribution.covariance_matrix.detach().numpy()

#     plt.plot(x_data, y_data * y_high_std + y_high_mean, "xk")


#     lengthscale =1.0; print("lengthscale:", model.covar_module.base_kernel.raw_lengthscale.detach().numpy().item())
#     alpha =1.0; print("alpha:", model.covar_module.raw_outputscale.detach().numpy().item())
#     sigma2 =1e-6; print("sigma2:", likelihood.noise.detach().numpy().item())

#     W = np.random.normal(size=(nFeatures, input_dim)) / lengthscale
#     b = np.random.uniform(low=0., high=(2 * np.pi), size=(nFeatures, 1))

#     theta = rff_sample_posterior_weights(x_data, y_data[:, 0], S_data, W, b, nFeatures=nFeatures, alpha=alpha, sigma2=sigma2)
    
#     def wrapper(x):
#         features = phi_rbf(x, W, b, alpha, nFeatures)
#         return theta @ features

#     return wrapper

# sa = sample_from_posterior_layer0(model.hidden_layer_0, x_t.shape[-1], likelihood=model.hidden_layer_likelihood_0, nFeatures=200, seed=0)

# #plt.plot(np.array([0., 0.09090909, 0.18181818, 0.27272727, 0.36363636,
# #                  0.45454545, 0.54545455, 0.63636364, 0.72727273, 0.81818182,
# #                  0.90909091, 1.]),
# #        np.array([-0.47065825, -0.63522399, -0.48852088, -0.24559148, -0.0570804 ,
# #                   0.16645165,  0.37990257,  0.31763058,  0.10252796,  0.45339102,
# #                   1.67323949,  2.70487143]) * y_high_std + y_high_mean, "Xk")

# x_t = np.linspace(-8., 8., 100)[:, None]
# plt.plot(x_t, sa(x_t) * y_high_std + y_high_mean)
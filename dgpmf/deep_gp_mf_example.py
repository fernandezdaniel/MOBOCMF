import numpy as np
import torch
import tqdm
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ExactGP
# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

from models.DeepGPMultifidelity import DeepGPMultifidelity
from gpytorch.mlls import DeepApproximateMLL

# from gpytorch.mlls import VariationalELBO
from mlls.VariationalELBOMultifidelity import VariationalELBOMultifidelity

def test_deep_gp_mf_example():
    np.random.seed(0)
    torch.manual_seed(0)

    # We define our fidelities

    LOW  = -1.0
    HIGH = 1.0
    LOC_MAXIMUM = 0.7862

    def obj_mf1(X):
        return (3-X)*np.sin(X*10) + X*1.2

    def obj_mf0(X):
        return 1.5*(3-X)*np.sin(X*10) + X*1.2 + 1.44

    NUM_INPUTS = 50
    NUM_FIDELITIES = 2 # 2
    l_input_fidelities = torch.randint(NUM_FIDELITIES, size=(NUM_INPUTS, 1))
    dgp_x = np.random.uniform(low=LOW, high=HIGH, size=(NUM_INPUTS, 1))
    dgp_y = np.array([obj_mf1(x_i) if i == 1 else obj_mf0(x_i) for i, x_i in zip(l_input_fidelities, dgp_x)]).reshape(dgp_x.shape)

    plt.plot(dgp_x[l_input_fidelities == 0], dgp_y[l_input_fidelities == 0], 'bo')
    plt.plot(dgp_x[l_input_fidelities == 1], dgp_y[l_input_fidelities == 1], 'ro')
    plt.show()

    dgp_train_x = torch.from_numpy(dgp_x).float()
    dgp_train_y = torch.from_numpy(dgp_y).float()

    dgp_train_dataset = TensorDataset(dgp_train_x, dgp_train_y, l_input_fidelities)
    dgp_train_loader = DataLoader(dgp_train_dataset, batch_size=10, shuffle=True)

    # GP_model = ExactGP()
    DGPMF_model = DeepGPMultifidelity(dgp_train_x.shape, num_hidden_layers=NUM_FIDELITIES) # comparar con un GP estandar

    # this is for running the notebook in our testing framework

    def train_dgp_mf(model, optimizer, train_x, train_loader, num_epochs=300, num_samples=1):
        mll = DeepApproximateMLL(VariationalELBOMultifidelity(model.likelihood, model, train_x.shape[-2]))

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")

        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            # n_iter = 0
            for (x_batch, y_batch, l_fidelities) in minibatch_iter:
                with gpytorch.settings.num_likelihood_samples(num_samples):
                    
                    optimizer.zero_grad()
                    
                    l_output = model.propagate(x_batch)
                    # output = l_output[l_fidelities]
                    loss = -mll(l_output, y_batch.T, l_fidelities=l_fidelities)
                    loss.backward()
                    optimizer.step()

                    minibatch_iter.set_postfix(loss=loss.item())


    optimizer = torch.optim.Adam([ {'params': DGPMF_model.parameters()}, ], lr=0.01)
    train_dgp_mf(DGPMF_model, optimizer, dgp_train_x, dgp_train_loader)



    num_samples_test = 25
    x_test = torch.from_numpy(np.linspace(-1, 1, 201)[:,None]).float()

    DGPMF_model.eval()
    samples = np.zeros((num_samples_test * 100, 201))
    for i in range(100):
        with gpytorch.settings.num_likelihood_samples(num_samples_test):
            predictive_means, predictive_variances = DGPMF_model.predict(x_test)
            samples[ (i * num_samples_test) : ((i + 1) * num_samples_test), : ] = np.random.normal(size = predictive_means.numpy().shape) * \
                 np.sqrt(predictive_variances.numpy()) + predictive_means.numpy()



    # We plot the results

    mean = np.mean(samples, 0)
    std = np.std(samples, 0)

    f, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.plot((dgp_train_x[l_input_fidelities == 0]).numpy(), (dgp_train_y[l_input_fidelities == 0]).numpy(), 'b*', label="Fidelity 0")
    ax.plot((dgp_train_x[l_input_fidelities == 1]).numpy(), (dgp_train_y[l_input_fidelities == 1]).numpy(), 'r*', label="Fidelity 1")
    ax.plot(np.linspace(-1, 1, 1000), obj_mf1(np.linspace(-1, 1, 1000)), 'g-', label="Ground truth")
    #ax.set_ylim([-3, 3])
    ax.set_xlim([LOW, HIGH])
    line, = ax.plot(x_test.numpy(), np.mean(samples, 0), 'b-')
    line.set_label('Predictive distribution DGP')
    line = ax.fill_between(x_test.numpy()[:,0], (mean + 2.0 * std), (mean - 2.0 * std), alpha=0.5)
    line.set_label('Confidence DGP')
    ax.legend()

    plt.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test_deep_gp_mf_example()

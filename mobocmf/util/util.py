
import os
import dill 
import torch
import numpy as np

def create_path(folder):
    # Check if the folder path exists
    if not os.path.exists(folder):
        # Create the folder path if it did not exist
        os.makedirs(folder)


def save_pickle(folder, filename, content):
    # Create the folder path
    create_path(folder)

    # Save the file
    with open(os.path.join(folder, filename), "wb") as fw:
        dill.dump(content, fw)

def read_pickle(folder, filename):

    with open(os.path.join(folder, filename), "rb") as fr:
        return dill.load(fr)
    
def triu_indices(n, offset=0):
    rows, cols = torch.triu_indices(n, n, offset=offset)
    indices = torch.stack((rows, cols), dim=0)
    return indices

def compute_dist(x):
    return torch.sum(x**2, 1, keepdims=True) - 2.0 * x.mm(x.T) + torch.sum(x**2, 1, keepdims=True).T


def preprocess_outputs(*args):

    # Important: DHL we use the same mean and standar deviation for each fidelity !!!

    y_mean = np.mean(np.vstack((args)))
    y_std = np.std(np.vstack((args)))

    y_train = []
    for y_in in args:
        y_out = (y_in - y_mean) / y_std
        y_train.append(torch.from_numpy(y_out).double())

    y_train.extend([y_mean, y_std])

    return y_train[:]

def preprocess_outputs_two_fidelities(y_low, y_high):

    # Important: DHL we use the same mean and standar deviation for each fidelity !!!

    y_mean = np.mean(np.vstack((y_high, y_low)))
    y_std = np.std(np.vstack((y_high, y_low)))

    y_high = (y_high - y_mean) / y_std
    y_train_high = torch.from_numpy(y_high).double()

    y_low = (y_low - y_mean) / y_std
    y_train_low = torch.from_numpy(y_low).double()

    return y_train_low, y_train_high, y_mean, y_std


def reset_random_state(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
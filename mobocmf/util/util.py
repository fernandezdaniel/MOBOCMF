
import os
import dill 
import torch

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


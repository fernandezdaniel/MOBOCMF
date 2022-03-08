import argparse

import numpy as np
from .dataset import get_dataset
import tensorflow as tf


def manage_experiment_configuration(args=None):

    if args is None:
        # Get parser arguments
        parser = get_parser()
        args = parser.parse_args()

    FLAGS = vars(args)
    # Manage Dataset
    args.dataset = get_dataset(args.dataset_name)

    if args.dtype == "float64":
        FLAGS["dtype"] = tf.float64

    return args


def check_data(X_train, y_train, verbose=1):
    if X_train.shape[0] != y_train.shape[0]:
        print("Labels and features differ in the number of samples")
        return

    # Compute data information
    n_samples = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)

    if verbose > 0:
        print("Number of samples: ", n_samples)
        print("Input dimension: ", input_dim)
        print("Label dimension: ", output_dim)
        print("Labels mean value: ", y_mean)
        print("Labels standard deviation: ", y_std)

    return n_samples, input_dim, output_dim, y_mean, y_std


def get_parser():
    """
    Defines and returns a parser for DGP experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples_train",
        type=int,
        default=1,
        help="Number of Monte Carlo samples of the posterior to "
        "use during training",
    )
    parser.add_argument(
        "--num_samples_test",
        type=int,
        default=200,
        help="Number of Monte Carlo samples of the posterior to "
        "use during inference",
    )
    parser.add_argument(
        "--genkern",
        type=str,
        default="RBF",
        help=(
            "Gaussian/Squared exponential/Radial Basis Function kernel"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Number of epochs to train de model",
    )
    parser.add_argument(
        "--gp_layers",
        type=int,
        default=[1],
        nargs="+",
        help="Sparse variational Gaussian Process layers structure",
    )
    parser.add_argument(
        "--final_layer_mu",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--final_layer_sqrt",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--inner_layers_mu",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--inner_layers_sqrt",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=200,
        help="Number of regression coefficients to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Training learning rate for the optimizer algorithm (e.g. Adam).",
    )
    parser.add_argument("--warmup", type=int, default=0)

    parser.add_argument("--shuffle", type=bool, default=True)

    parser.add_argument("--show", dest="show", action="store_true")
    parser.set_defaults(show=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=2147483647,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
    )
    parser.add_argument(
        "--split",
        default=None,
        type=int,
    )
    parser.add_argument("--name_flag", default="", type=str)

    return parser

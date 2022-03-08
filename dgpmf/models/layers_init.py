import numpy as np
import tensorflow as tf

from dgpmf.models.layers import SVGPLayer
from dgpmf.kernels.matern import MaternKernel
from dgpmf.kernels.rbf import RBFKernel


class LinearProjection:
    def __init__(self, matrix):
        """
        Encapsulates a linear projection defined by a Matrix

        Parameters
        ----------
        matrix : np.darray of shape (N, M)
                 Contains the linear projection
        """
        self.P = matrix

    def __call__(self, inputs):
        """
        Applies the linear transformation to the given input.
        """
        return inputs @ self.P.T


def init_layers(
    X,
    output_dim,
    gp_layers,
    genkern,
    num_inducing_points,
    dtype,
    final_layer_mu,
    final_layer_sqrt,
    inner_layers_sqrt,
    inner_layers_mu,
    **kwargs
):
    """
    Creates the Deep Gaussian Process layers using the given
    information. If the dimensionality is reducen between layers,
    these are created with a mean function that projects the data
    to their maximum variance projection (PCA).
    If several projections are made, the first is computed over the
    original data, and, the following are applied over the already
    projected data.
    Parameters
    ----------
    X : tf.tensor of shape (num_data, data_dim)
        Contains the input features.
    output_dim : int
                 Number of output dimensions of the model.
    gp_layers : integer or list of integers
                 Indicates the number of GP layers to use. If
                 an integer is used, as many layers as its value
                 are created, with output dimension output_dim.
                 If a list is given, layers are created so that
                 the dimension of the data matches these values.
                 For example, inner_dims = [10, 3] creates 3
                 layers; one that goes from data_dim features
                 to 10, another from 10 to 3, and lastly from
                 3 to output_dim.
    genkern : string
              Indicates the kernel function to use, can be: Matern52, Matern32 or RBF.
    num_inducing_poitns : integer
                          Number of inducing points to use.
    dtype : data-type
                The dtype of the layer's computations and weights.
    """

    # Create GP layers. If integer, replicate input dimension
    if len(gp_layers) == 1:
        gp_layers = [X.shape[1]] * (gp_layers[0] - 1)
        dims = [X.shape[1]] + gp_layers + [output_dim]
    # Otherwise, append thedata dimensions to the array.
    else:
        if gp_layers[-1] != output_dim:
            raise RuntimeError(
                "Last gp layer does not correspond with data label"
            )
        dims = [X.shape[1]] + gp_layers

    # Initialize layers array
    layers = []
    # We maintain a copy of X, where each projection is applied. That is,
    # if two data reductions are made, the matrix of the second is computed
    # using the projected (from the first projection) data.
    X_running = np.copy(X)
    for (i, (dim_in, dim_out)) in enumerate(zip(dims[:-1], dims[1:])):

        # Last layer has no transformation
        if i == len(dims) - 2:
            mf = None
            q_mu_initial_value = final_layer_mu
            q_sqrt_initial_value = final_layer_sqrt

        # No dimension change, identity matrix
        elif dim_in == dim_out:
            mf = lambda x: x
            q_mu_initial_value = inner_layers_mu
            q_sqrt_initial_value = inner_layers_sqrt

        # Dimensionality reduction, PCA using svd decomposition
        elif dim_in > dim_out:
            _, _, V = np.linalg.svd(X_running, full_matrices=False)

            mf = LinearProjection(V[:dim_out, :].T)
            # Apply the projection to the running data,
            X_running = X_running @ V[:dim_out].T

        else:
            raise NotImplementedError(
                "Dimensionality augmentation is not handled currently."
            )

        if genkern == "Matern52":
            kern = MaternKernel(length_scale=0.2, noise_scale=1e-6, output_scale=1.0, nu=2.5)
        elif genkern == "Matern32":
            kern = MaternKernel(length_scale=0.2, noise_scale=1e-6, output_scale=1.0, nu=1.5)
        elif genkern == "Gaussian" or genkern == "RBF":
            kern = RBFKernel(length_scale=0.2, noise_scale=1e-6, output_scale=1.0)
        else:
            raise ValueError("The genkern:", genkern, "is not implemented, please choose an implemented kernel (Matern52, Matern32 or RBF).")
        
        Z_shape = (num_inducing_points,) + (X.shape[1:])
        # Create layer
        layers.append(
            SVGPLayer(
                kern,
                Z_shape=Z_shape,
                input_dim=dim_in,
                output_dim=dim_out,
                mean_function=mf,
                q_mu_initial_value=q_mu_initial_value,
                q_sqrt_initial_value=q_sqrt_initial_value,
                dtype=dtype,
            )
        )

    return layers

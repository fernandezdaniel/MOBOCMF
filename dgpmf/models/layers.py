
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..utils import reparameterize
import dgpmf.settings as settings


class Layer(tf.keras.layers.Layer):
    def __init__(self, input_dim=None, dtype=None, seed=0):
        """
        A base class for GP layers. Basic functionality for multisample
        conditional and input propagation.

        Parameters
        ----------
        input_dim : int
                    Input dimension
        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        seed : int
               integer to use as seed for randomness.
        """
        super().__init__(dtype=dtype)
        self.seed = seed
        self.build(input_dim)

    def train_mode(self):
        self.training = True

    def eval_mode(self):
        self.training = False

    def conditional_ND(self, X, full_cov=False):
        """
        Computes the conditional probability

        Parameters
        ----------
        X : tf.tensor of shape (N, D)
            Contains the input locations.

        full_cov : boolean
                   Wether to use full covariance matrix or not.
                   Determines the shape of the variance output.

        Returns
        -------
        mean : tf.tensor of shape (N, output_dim)
               Contains the mean value of the distribution for each input

        var : tf.tensor of shape (N, output_dim) or (N, N, output_dim)
              Contains the variance value of the distribution for each input
        """
        raise NotImplementedError

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior.
        """
        raise NotImplementedError

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        raise NotImplementedError

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample.
        Adds input propagation if necessary.

        Parameters
        ----------
        X : tf.tensor of shape (N, D_in)
            Input locations.

        z : tf.tensor of shape (N, D)
            Contains a sample from a Gaussian distribution, ideally from a
            standardized Gaussian.

        full_cov : boolean
                   Wether to compute or not the full covariance matrix or just
                   the diagonal.

        Returns
        -------
        samples : tf.tensor of shape (S, N, self.output_dim)
                  Samples from a Gaussian given by mean and var. See below.
        mean : tf.tensor of shape (S, N, self.output_dim)
               Stacked tensor of mean values from conditional_ND applied to X.
        var : tf.tensor of shape (S, N, self.output_dim or
                                  S, N, N, self.output_dim)
              Stacked tensor of variance values from conditional_ND applied to
              X.
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = self.output_dim

        mean = tf.reshape(mean, (S, N, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, N, D))

        # If no sample is given, generate it from a standardized Gaussian
        if z is None:
            z = tf.random.normal(
                shape=tf.shape(mean), seed=self.seed, dtype=settings.float_type
            )
        # Apply re-parameterization trick to z
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        return samples, mean, var


class SVGPLayer(Layer):

    class InducingPoints():
        """
        Real-space inducing points
        """

        def __init__(self, Z, dtype=tf.float64):
            """
            :param Z: the initial positions of the inducing points, size M x D
            """
            super().__init__()
            self.Z = tf.Variable(Z, name="Z")
            self.dtype = dtype

        def __len__(self):
            return self.Z.shape[0]

        @tf.function # DFS: If uncomment @tf.function, the TensorFlow raise some Warnings
        def Kuu(self, kern, jitter=0.0):
            Kzz = kern(self.Z)
            Kzz += jitter * tf.eye(len(self), dtype=self.dtype)
            return Kzz

        @tf.function # DFS: If uncomment @tf.function, the TensorFlow raise some Warnings
        def Kuf(self, kern, Xnew):
            Kzx = kern(self.Z, Xnew)
            return Kzx



    def __init__(
        self,
        kern,
        Z_shape,
        output_dim,
        input_dim,
        q_mu_initial_value,
        q_sqrt_initial_value,
        mean_function=None,
        dtype=tf.float64,
        seed=0,
    ):
        """
        A Sparse variational Gaussian Process layer.

        Parameters
        ----------
        kern : The kernel for the layer (input_dim = D_in)

        num_inducing_points: Number of inducing points to use

        output_dim : int
                      The number of independent SVGP in this layer.
                      More precisely, q_mu has shape (S, output_dim)

        input_dim : int
                    Dimensionality of the given features. Used to
                    pre-fix the shape of the different layers of the model.

        seed : int
               Seed to be used for the randomness of the layer

        mean_function : callable
                        Mean function added to the model. If no mean function
                        is specified, no value is added.

        dtype : data-type
                The dtype of the layer's computations and weights.
                Refer to tf.keras.layers.Layer for more information.
        """
        super().__init__(dtype=dtype, input_dim=input_dim, seed=seed)
        self.num_inducing = Z_shape[0]
        
        # Regression Coefficients prior mean
        self.q_mu = tf.Variable(
            np.ones((self.num_inducing, output_dim)) * q_mu_initial_value,
            name="q_mu",
        )

        # Define Regression coefficients deviation using tiled triangular
        # identity matrix
        # Shape (num_coeffs, num_coeffs)
        q_var = np.eye(self.num_inducing) * q_sqrt_initial_value
        # Replicate it output_dim times
        # Shape (output_dim, num_coeffs, num_coeffs)
        q_var = np.tile(q_var[None, :, :], [output_dim, 1, 1])
        # Create tensor with triangular representation.
        # Shape (output_dim, num_coeffs*(num_coeffs + 1)/2)
        q_sqrt_tri_prior = tfp.math.fill_triangular_inverse(q_var)
        self.q_sqrt_tri = tf.Variable(q_sqrt_tri_prior, name="q_sqrt_tri")

        Z = np.random.normal(size=Z_shape)
        self.feature = self.InducingPoints(Z, dtype=self.dtype)
        self.kern = kern
        # If no mean function is given, constant 0 is used
        self.mean_function = mean_function

        # Verticality of the layer
        self.output_dim = tf.constant(output_dim, dtype=tf.int32)

        self.Ku = None
        self.Lu = None
        self.Ku_tiled = None
        self.Lu_tiled = None
        self.needs_build_cholesky = True

    @tf.function  # DFS: Do we need the tf.function decorator here? (If I put it, an expcetion occurs)
    def build_cholesky_if_needed(self):
        # make sure we only compute this once
        if self.needs_build_cholesky:
            Ku = self.feature.Kuu(self.kern, jitter=settings.jitter)
            Lu = tf.linalg.cholesky(Ku)
            Ku_tiled = tf.tile(Ku[None, :, :], [self.output_dim, 1, 1])
            Lu_tiled = tf.tile(Lu[None, :, :], [self.output_dim, 1, 1])
            self.needs_build_cholesky = False

            return Ku, Lu, Ku_tiled, Lu_tiled

        else:
            return self.Ku, self.Lu, self.Ku_tiled, self.Lu_tiled

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """

        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [tf.reshape(m, [S, N, self.output_dim]) for m in [mean, var]]

    @tf.function # DFS: Do we need the tf.function decorator here? (If I put it, an expcetion occurs)
    def conditional_ND(self, X, full_cov=False):
        """


        Parameters:
        -----------
        X : tf.tensor of shape (N, D)
            Contains the input locations.

        full_cov : boolean
                   Wether to use full covariance matrix or not.
                   Determines the shape of the variance output.

        Returns:
        --------
        mean : tf.tensor of shape (N, output_dim)
               Contains the mean value of the distribution for each input

        var : tf.tensor of shape (N, output_dim) or (N, N, output_dim)
              Contains the variance value of the distribution for each input
        """

        self.Ku, self.Lu, self.Ku_tiled, self.Lu_tiled = self.build_cholesky_if_needed()

        Kuf = self.feature.Kuf(self.kern, X)

        A = tf.linalg.triangular_solve(self.Lu, Kuf, lower=True)
        A = tf.linalg.triangular_solve(tf.transpose(self.Lu), A, lower=False)

        mean = tf.linalg.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, :, :], [self.output_dim, 1, 1])

        SK = -self.Ku_tiled

        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        SK += tf.linalg.matmul(q_sqrt, q_sqrt, transpose_b=True)

        B = tf.linalg.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.linalg.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kern(X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.math.reduce_sum(A_tiled * B, 1)
            Kff = self.kern.Kdiag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        # Add mean function
        if self.mean_function is not None:
            mean = mean + self.mean_function(X)

        return mean, var


    @tf.function
    def KL(self):
        """
        Computes the KL divergence from the variational distribution of
        the linear regression coefficients to the prior.

        That is from a Gaussian N(q_mu, q_sqrt) to N(0, I).
        Uses formula for computing KL divergence between two
        multivariate normals, which in this case is:

        KL = 0.5 * ( tr(q_sqrt^T q_sqrt) +
                     q_mu^T q_mu - M - log |q_sqrt^T q_sqrt| )
        """

        D = tf.cast(self.output_dim, dtype=self.dtype)

        # Recover triangular matrix from array
        q_sqrt = tfp.math.fill_triangular(self.q_sqrt_tri)
        # Constant dimensionality term
        KL = -0.5 * D * self.num_inducing
        # Log of determinant of covariance matrix.
        # Det(Sigma) = Det(q_sqrt q_sqrt^T) = Det(q_sqrt) Det(q_sqrt^T)
        #            = prod(diag_s_sqrt)^2
        KL -= tf.reduce_sum(
            tf.math.log(tf.math.abs(tf.linalg.diag_part(q_sqrt)))
        )

        # Trace term
        KL += 0.5 * tf.reduce_sum(tf.math.square(self.q_sqrt_tri))

        # Mean term
        KL += 0.5 * tf.reduce_sum(tf.math.square(self.q_mu))

        return KL

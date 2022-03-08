import tensorflow as tf
import math
from dgpmf.kernels.kernel import Kernel

class MaternKernel(Kernel):
    r"""
    Computaiton of matern kernel
       \begin{equation*}
          k_{\text{Matern}}(\mathbf{x_1}, \mathbf{x_2}) = \frac{2^{1 - \nu}}{\Gamma(\nu)}
          \left( \sqrt{2 \nu} d \right) K_\nu \left( \sqrt{2 \nu} d \right)
       \end{equation*}
    """

    def __init__(self, length_scale, noise_scale, output_scale, nu=1.5, jitter=1e-3, num_dims=1):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__(jitter)
        self.nu = nu
        self.lengthscale = tf.Variable([length_scale]* num_dims, dtype = self.dtype, name="lengthscale")
        self.output_scale = tf.Variable([output_scale], dtype=self.dtype, name="output_scale")
        #self.variance = tf.Variable([np.exp(self.output_scale)], dtype=self.dtype, name="variance")
        self.whitenoise_scale = tf.constant([noise_scale], dtype=self.dtype) # tf.Variable([noise_scale], dtype=self.dtype, name="whitenoise_scale") # I had to change this var to a cte because of the warning issues: """WARNING:tensorflow:Gradients do not exist for variables ['whitenoise_scale:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss`argument?"""
        self.jitter = tf.constant([jitter], dtype=self.dtype)

    #@tf.function # DFS: If uncomment @tf.function, the TensorFlow raise some Warnings
    def call(self, x1, x2 = None):
        """
        Computation of kernel
        """

        if x2 is None:
            x2 = x1
            if (len(tf.shape(x1)) == 2):
                eye_matrix = tf.eye(tf.shape(input=x1)[ 0 ], dtype = self.dtype)
            else:
                eye_matrix = tf.eye(tf.shape(input=x1)[ 1 ], batch_shape = [tf.shape(input=x1)[ 0 ]], dtype = self.dtype)
                
            white_noise = (self.jitter + tf.exp(self.whitenoise_scale)) * eye_matrix
        else:
            white_noise = 0.0
        
        if len(tf.shape(x1)) > len(tf.shape(x2)):
            x2 = tf.expand_dims(x2, 1)
        elif len(tf.shape(x2)) > len(tf.shape(x1)):
            x1 = tf.expand_dims(x1, 1)

        x1 = x1 / tf.exp(self.lengthscale)
        x2 = x2 / tf.exp(self.lengthscale)
        # pairwise distance computation
        distance = self._compute_distance(x1, x2)
        
        # multiply by term
        exp_c = tf.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            # Matern 1/2
            const_c = 1
        elif self.nu == 1.5:
            # Matern 3/2
            const_c = 1.0 + math.sqrt(3) * distance
        elif self.nu == 2.5:
            # Matern 5/2
            const_c = math.sqrt(5) * distance + 1.0 + 5.0 / 3.0 * distance ** 2


        return tf.exp(self.output_scale) * const_c * exp_c + white_noise


    def get_params(self):
        """
        get learnable parameters
        """
        return [ self.lengthscale, self.output_scale, self.whitenoise_scale ]

    @tf.function
    def get_var_points(self, data_points):
        return  tf.ones([ tf.shape(input=data_points)[ 0 ] ], dtype=self.dtype) * tf.exp(self.output_scale) + \
                (self.jitter + tf.exp(self.whitenoise_scale))
    def get_var_each_point(self, data_inputs):
        return tf.exp(self.output_scale) + (self.jitter + tf.exp(self.whitenoise_scale))

    #@tf.function # DFS: If uncomment @tf.function, the TensorFlow raise some Warnings
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(tf.exp(self.output_scale)))

    @tf.function
    def _compute_distance(self, X1, X2):
        """
        Computes pairwise Euclidean distances between x1 and x2
        Can compute batch distances when dimensionality = 3
        Args:
          x1,    [?,m,d] matrix
          x2,    [?,n,d] matrix
        Returns:
          covar,    [?,m,n] Euclidean distances
        """

        value = tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X1), axis=-1), -1)
        
        if len(tf.shape(X2)) == 2:
            value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X2), axis=-1), -1))
        else:
            value2 = tf.transpose(a=tf.expand_dims(tf.reduce_sum(input_tensor=tf.square(X2), axis=-1), -1), perm=[0, 2, 1])
        
        covar = tf.maximum(value - 2 * tf.matmul(X1, X2, False, True) + value2, 0.0)
        
        covar = tf.sqrt(tf.maximum(covar, 1e-40))
        
        return covar

import tensorflow as tf
import numpy as np


class NoiseSampler:
    def __init__(self, seed):
        """
        Generates noise samples.

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        self.seed = seed

    def call(self):
        """
        Returns sampled noise values.
        """
        raise NotImplementedError


class GaussianSampler(NoiseSampler):
    def __init__(self, seed):
        """
        Generates noise samples from a Standar Gaussian distribution N(0, 1).

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        super().__init__(seed)
        self.rng = tf.random.Generator.from_seed(self.seed)

    def reset_seed(self):
        self.rng.reset_from_seed(self.seed)

    def __call__(self, size):
        """
        Returns sampled noise values os the given size or shape.

        Parameters:
        -----------
        size : int or np.darray
               Indicates the desired shape/size of the sample to generate.

        Returns:
        --------
        samples : np.darray of shape (size)
                  A sample from a Gaussian distribution N(0, I).

        """
        return self.rng.normal(size, dtype=tf.float64)


class UniformSampler(NoiseSampler):
    def __init__(self, seed):
        """
        Generates noise samples from a Uniform distribution.

        Parameters:
        -----------
        seed : int
               Integer value used to generate reproducible results.

        """
        super().__init__(seed)
        self.rng = tf.random.Generator.from_seed(self.seed)

    def reset_seed(self):
        self.rng.reset_from_seed(self.seed)

    def __call__(self, size):
        """
        Returns sampled noise values os the given size or shape.

        Parameters:
        -----------
        size : int or np.darray
               Indicates the desired shape/size of the sample to generate.

        Returns:
        --------
        samples : np.darray of shape (size)
                  A sample from a Gaussian distribution N(0, I).

        """
        return self.rng.uniform(size, dtype=tf.float64)


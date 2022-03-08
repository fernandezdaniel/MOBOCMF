
import abc
import tensorflow as tf
import dgpmf.settings as settings

class Kernel(tf.keras.layers.Layer):
    '''
    Generic Kernel class
    '''
    __metaclass__ = abc.ABCMeta
    _jitter = 1e-10
    
    def __init__(self, jitter=1e-10):
        super(Kernel, self).__init__(dtype=settings.tf_float_type)
        self._jitter = jitter
   
    @abc.abstractmethod
    def call(self, X, X2=None):
        raise NotImplementedError("Subclass should implement this.")
    
    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")
        
    @classmethod
    def jitter(self):
        return self._jitter

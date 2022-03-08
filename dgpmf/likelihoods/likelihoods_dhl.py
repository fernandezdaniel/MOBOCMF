import abc
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import math
import tensorflow_probability as tfp
import dgpmf.settings as settings

class Likelihood(tf.keras.layers.Layer):
    '''
    Class representing a likelihood.
    Any subclass must implement the methods variational_expectations and negative_log_likelihood
    '''
    
    def __init__(self):
        super(Likelihood, self).__init__(dtype=settings.tf_float_type)

    def metrics(self):
        return []

    def update_metrics(self):
        pass

    @abc.abstractmethod
    def variational_expectations(self, means, variances, outputs):
        '''
        Computes E[log p(y|f)] = ∫ log(p(y|f)) q(f) df
        '''
        raise NotImplementedError("Subclass should implement this.")

    def call(self, means, variances):
        '''
        Computes means and variances of E_q p(y|f)

        '''
        return means, tf.transpose(variances)

    @abc.abstractmethod
    def negative_log_likelihood(self, means, variances, outputs):
        '''
        Computes log E[p(y|f)] = log ∫ p(y|f)q(f) df
        '''
        raise NotImplementedError("Subclass should implement this.")


class GaussianLikelihood(Likelihood):
    '''
    Implements Gaussian likelihood for regression
    '''
    def __init__(self, init_var = 0.1):
        super(GaussianLikelihood, self).__init__()

        self.log_var_noise = tf.Variable(np.log(init_var), dtype = self.dtype, name = "log_var_noise") # // \sigma^{2}

        self.rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")

        self.nll_metric = tf.keras.metrics.Mean(name="nll")

    @property
    def metrics(self):
        return [self.rmse_metric, self.nll_metric]

    @tf.function
    def update_metrics(self, y, mean_pred, var_pred):
        self.rmse_metric.update_state(y, mean_pred)

        nll = self.negative_log_likelihood(mean_pred, var_pred, y)

        self.nll_metric.update_state(nll)

    @tf.function
    def variational_expectations(self, means, variances, outputs):
        return -0.5 * tf.math.log(2 * tf.constant(np.pi, dtype = self.dtype)) - 0.5 * self.log_var_noise - 0.5 / tf.exp(self.log_var_noise) * \
            (outputs**2 - 2.0 * outputs * tf.transpose(a=means) + tf.transpose(a=means**2 + variances))
    
    @tf.function
    def call(self, means, variances):
        return means, tf.transpose(a=variances) + tf.exp(self.log_var_noise)

    @tf.function
    def negative_log_likelihood(self, means, variances, outputs):
        f1 = tf.math.log(variances) / 2
        f2 = (outputs - means)**2 / (2 * variances)
        f3 = tf.math.log(2 * tf.constant(math.pi, dtype = self.dtype)) / 2

        return tf.reduce_mean(f1 + f2 + f3)
        

class BernoulliLikelihood(Likelihood):
    def __init__(self, ngh = 200):
        super(BernoulliLikelihood, self).__init__()

        self.ngh = ngh

        self.HpolyX, self.HW = self.herm_gauss(ngh)

        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

        self.nll_metric = tf.keras.metrics.Mean(name="nll")

        self.normal_dist = tfp.distributions.Normal(loc = tf.cast(0, dtype= self.dtype),
                                                    scale= tf.cast(1, dtype= self.dtype))

    #@tf.function
    def herm_gauss(self, n):
        """
        ∫exp(-x**2)f(x)dx = \sum_{i=1}^{n} hw_i/sqrt(pi) h(sqrt(2)*x_i* stddev + mean)
        output hw, x

        """
        hpoly, hw = np.polynomial.hermite.hermgauss(n)

        hpoly = hpoly * np.sqrt(2)

        hw = hw / np.sqrt(np.pi)

        hw /= np.linalg.norm(hw, ord=1, keepdims=True)

        return tf.cast(tf.reshape(hpoly, [len(hpoly), 1]), dtype=settings.tf_float_type), \
               tf.cast(tf.reshape(hw, [len(hw), 1]), dtype=settings.tf_float_type)

    #@tf.function
    def log_likelihood(self, pred, y):
        # tf.reduce_sum(self.log_Gauss_hermitQuadrature(self._quadrature_log_prob, means, variances, outputs), axis=-1)

        y = tf.squeeze(y)

        p = pred

        p = tf.cast(y < 0, self.dtype) * (1 - p) + tf.cast(y >= 0, self.dtype) * p

        return tf.math.log(p)


    #@tf.function
    def negative_log_likelihood(self, pred, outputs):
        return - tf.reduce_mean(self.log_likelihood(pred, outputs))

    #@tf.function
    def variational_expectations(self, means, var, y):
        quadrature_result = self.Gauss_hermitQuadrature(self._quadrature_log_prob, means, var, y)
        return quadrature_result #tf.squeeze(quadrature_result, axis=-1)

    #@tf.function
    def _compute_log_bernoulli(self, outputs, p):
        return tf.math.log(tf.where(tf.equal(outputs, 1), p, 1-p))

    #@tf.function
    def _quadrature_log_prob(self, dist, y):
        """
        The log probability density log p(y|f)
        """
        #probit = self.cdf_normal(dist) # P(y|f)
        #log_prob = self._compute_log_bernoulli(y, probit) # log p(y|f)

        dist = dist * y
        log_prob = self.normal_dist.log_cdf(dist)

        log_prob = tf.reduce_sum( log_prob, axis=[1])
        return log_prob

    #@tf.function
    def call(self, means, var):
        probit = self.normal_dist.cdf(means / tf.sqrt(1 + var))
        return probit, probit - probit**2

    #@tf.function
    def shift_means(self, means, var):
        """
        out = mean + stddev*Hpoly
        """
        means = tf.expand_dims(means, 0)
        stddev = tf.expand_dims(tf.sqrt(var), 0)
        return means + stddev*self.HpolyX

    #@tf.function
    def Gauss_hermitQuadrature(self, fun, means, var, y):
        """
        E[f] = ∫ f(x) p(x) dx  = ∫exp(-x**2)f(x)dx = \sum_{i=1}^{n} hw_i/sqrt(pi) h(sqrt(2)*x_i* stddev + mean)
        """
        shifted_locations = self.shift_means(means, var)
        return tf.reduce_sum(fun(shifted_locations, y) * self.HW[:,0], axis=0)

    # @tf.function
    def log_Gauss_hermitQuadrature(self, fun, means, var, y):
        """
        log E[exp[f]] = log ∫exp[f]p(x)dx
        """
        shifted_locations = self.shift_means(means, var)
        logW = tf.math.log(self.HW)
        return tf.reduce_logsumexp(fun(shifted_locations, y) + logW, axis=0)


    @property
    def metrics(self):
        return [self.acc_metric, self.nll_metric]

    #@tf.function
    def update_metrics(self, y, mean_pred, var_pred):
        p = tf.cast(tf.where(mean_pred < 0.5, -1.0, 1.0), dtype= settings.tf_float_type)
        self.acc_metric.update_state(y, p)

        nll = self.negative_log_likelihood(mean_pred, y)
        self.nll_metric.update_state(nll)


class RobustLikelihood(Likelihood):
    def __init__(self, epsilon = 1e-3):
        super(RobustLikelihood, self).__init__()

        self.epsilon = tf.cast(epsilon, self.dtype)

        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

        self.nll_metric = tf.keras.metrics.Mean(name="nll")

        self.normal_dist = tfp.distributions.Normal(loc = tf.cast(0, dtype= self.dtype),
                                                    scale= tf.cast(1, dtype= self.dtype))

    @tf.function
    def log_likelihood(self, pred, y):
        # log E_q [ p(y|f(x)] = (1 - e) norm_cdf (m_f / sqrt(v_f))  + e  (1 - norm_cdf( m_f / sqrt(v_f)))
        # tf.math.log((1 - self.epsilon) * self.normal_dist.cdf(p) + self.epsilon * (1 - self.normal_dist.cdf(p)))

        y = tf.squeeze(y)

        p = pred

        p = tf.cast(y < 0, self.dtype) * (1 - p) + tf.cast(y >= 0, self.dtype) * p

        return tf.math.log(p)

        
    @tf.function
    def negative_log_likelihood(self, pred, outputs):
        return - tf.reduce_mean(self.log_likelihood(pred, outputs))

    @tf.function
    def variational_expectations(self, means, var, y):
        # E_q [ log p (y|f(x)] = log(1 - e) * prob(y * f(x) > 0 ) + log(e) * prob(y * f(x) <= 0)
        y = tf.squeeze(y)
        means = tf.squeeze(means)
        var = tf.squeeze(var)

        p = y * means / tf.sqrt(var)

        return tf.math.log(1 - self.epsilon) * self.normal_dist.cdf(p) + tf.math.log(self.epsilon) * (1 - self.normal_dist.cdf(p))

    @tf.function
    def call(self, means, var):
        probit = self.normal_dist.cdf(means / tf.sqrt(var))
        mean = (1 - self.epsilon) * probit + self.epsilon * (1 - probit)
        return mean, mean - mean**2

    @property
    def metrics(self):
        return [self.acc_metric, self.nll_metric]

    @tf.function
    def update_metrics(self, y, mean_pred, var_pred):
        p = tf.cast(tf.where(mean_pred < 0.5, -1.0, 1.0), dtype=settings.tf_float_type)
        self.acc_metric.update_state(y, p)

        nll = self.negative_log_likelihood(mean_pred, y)
        self.nll_metric.update_state(nll)


class BernoulliLikelihood_sigmoid(Likelihood):
    def __init__(self, ngh=50):
        super(BernoulliLikelihood_sigmoid, self).__init__()

        self.ngh = ngh

        self.HpolyX, self.HW = self.herm_gauss(ngh)

        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

        self.nll_metric = tf.keras.metrics.Mean(name="nll")

        self.sigmoid = tf.nn.sigmoid

    @tf.function
    def herm_gauss(self, n):
        """
        @return:
        ∫exp(-x**2)f(x)dx = \sum_{i=1}^{n} hw_i/sqrt(pi) h(sqrt(2)*x_i* stddev + mean)
        output hw, x
        """
        hpoly, hw = np.polynomial.hermite.hermgauss(n)

        hpoly = hpoly * np.sqrt(2)

        hw = hw / np.sqrt(np.pi)

        hw /= np.linalg.norm(hw, ord=1, keepdims=True)

        return tf.cast(tf.reshape(hpoly, [len(hpoly), 1]), dtype=settings.tf_float_type), \
               tf.cast(tf.reshape(hw, [len(hw), 1]), dtype=settings.tf_float_type)

    @tf.function
    def log_likelihood(self, pred, y):
        """
        Computes log E[p(y|f)] = log ∫ p(y|f)q(f) df
        """
        y = tf.squeeze(y)
        p = pred

        p = tf.cast(y < 0, self.dtype) * (1 - p) + tf.cast(y >= 0, self.dtype) * p

        return tf.math.log(p)

        # means = tf.squeeze(means)
        # var = tf.squeeze(var)
        # #return tf.math.log(self._Gauss_hermitQuadrature(self._quadrature_prob, means, var, y))
        # return self._log_Gauss_hermitQuadrature(self.__quadrature_log_prob, means, var, y)


    def __quadrature_prob(self, dist, y):
        """
        The probability density p(y|f)
        """
        dist = dist * y
        prob = self.sigmoid(dist)
        return prob

    @tf.function
    def negative_log_likelihood(self, pred, outputs):
        return - tf.reduce_mean(self.log_likelihood(pred, outputs))

    @tf.function
    def variational_expectations(self, means, var, y):
        """
        Computes E[log p(y|f)] = ∫ log(p(y|f)) q(f) df
        """
        quadrature_result = self._Gauss_hermitQuadrature(self.__quadrature_log_prob, means, var, y)
        return quadrature_result


    @tf.function
    def __quadrature_log_prob(self, dist, y):
        """
        The log probability density log p(y|f)
        """
        dist = dist * y
        log_prob = tf.math.log(self.sigmoid(dist))
        return log_prob


    @tf.function
    def call(self, means, var):
        """
        @param means:
        @param var:
        @return:
            likelihood
        """
        probit = self._Gauss_hermitQuadrature_Post(means, var)
        return probit, probit - probit ** 2

    @tf.function
    def __shift_means(self, means, var):
        """
        out = mean + stddev*Hpoly
        """
        means = tf.expand_dims(means, 0)
        stddev = tf.expand_dims(tf.sqrt(var), 0)
        return means + stddev * self.HpolyX

    @tf.function
    def _Gauss_hermitQuadrature(self, fun, means, var, y):
        """
        E[f] = ∫ f(x) p(x) dx  = ∫exp(-x**2)f(x)dx = \sum_{i=1}^{n} hw_i/sqrt(pi) h(sqrt(2)*x_i* stddev + mean)
        """
        shifted_locations = self.__shift_means(means, var)
        return tf.reduce_sum(tf.matmul(tf.transpose(self.HW), fun(shifted_locations, y)), axis = 0)
        #return tf.reduce_sum(fun(shifted_locations, y) * self.HW[:, 0], axis=0)

    @tf.function
    def _Gauss_hermitQuadrature_Post(self, means, var):
        """
        @return:
        E_f*[exp(f*)/(1+exp(f*)] = ∫ exp(f*)/(1+exp(f*)) p(f*|y) df*
        """
        shifted_locations = self.__shift_means(means, var)
        return tf.reduce_sum(self.HW*self.sigmoid(shifted_locations), axis = 0)

    def _log_Gauss_hermitQuadrature(self, fun, means, var, y):
        """
        log E[exp[f]] = log ∫exp[f]p(x)dx
        """
        shifted_locations = self.__shift_means(means, var)
        logW = tf.math.log(self.HW)
        return tf.reduce_logsumexp(fun(shifted_locations, y) + logW, axis=0)

    @property
    def metrics(self):
        return [self.acc_metric, self.nll_metric]

    @tf.function
    def update_metrics(self, y, mean_pred, var_pred):
        """
        Update metrics
        """
        p = tf.cast(tf.where(mean_pred < 0.5, -1.0, 1.0), dtype=settings.tf_float_type)
        self.acc_metric.update_state(y, p)

        nll = self.negative_log_likelihood(mean_pred, y)
        self.nll_metric.update_state(nll)
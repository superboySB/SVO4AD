import numpy as np
import math
from scipy.stats import vonmises

np.seterr(invalid='ignore')


class HistogramSVO(object):
    def __init__(self,
                 rollout_step,
                 n_discrete=91,
                 bound=None):
        """
        Histogram Filter
        params:
            n_discrete: num of histogram
            bound: bound of the histogram value
        """
        if bound is None:
            # bound = [-np.pi / 2, np.pi / 2]
            bound = [0, np.pi / 2]
        self.n_discrete = n_discrete
        self.histograms = np.linspace(bound[0], bound[1], num=n_discrete)
        self.bound = bound
        self.weights = np.ones([n_discrete, ]) / n_discrete

        self.svo_error_history = []
        self.mean, self.std, self.error = self._get_mean_std_error()
        self.sample_svo = 0  # 利己保守主义
        for _ in range(rollout_step):
            self.svo_error_history.append([self.sample_svo, self.error])

    def update(self, posteriori_probabilities):
        self.weights = self.weights * self._vonmisesvariate()
        self.weights = self.weights * posteriori_probabilities
        self.weights = self.weights / self.weights.sum()
        self.mean, self.std, self.error = self._get_mean_std_error()
        self.sample_svo = self._sample()
        self.svo_error_history.append([self.sample_svo, self.error])

    def _sample(self):
        sim_sample = np.random.choice(self.histograms, p=self.weights)
        return sim_sample

    def _get_mean_std_error(self, ):
        """return the mean and variance of SVO
        """
        mean = (self.histograms * self.weights).sum()
        std = ((self.histograms - mean) ** 2 * self.weights).sum()
        var = math.sqrt(std)
        error = np.random.normal(mean, var)
        return mean, std, error

    def _vonmisesvariate(self):
        rv = vonmises(self.std)
        vons = []

        for i in range(self.n_discrete):
            vons.append(rv.pdf(self.histograms[i]))
        return np.array(vons)

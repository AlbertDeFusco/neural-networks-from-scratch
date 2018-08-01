import numpy as np

class CrossEntropy(object):
    def __call__(self, y, T):
        return -(T * np.log(y) + (1 - T) * np.log(1 - y)).mean()

    def gradient(self, y, T):
        return (y - T) / y.shape[0]

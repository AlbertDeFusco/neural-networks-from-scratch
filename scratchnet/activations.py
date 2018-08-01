import numpy as np

class Relu(object):
    def __call__(self, a):
        return np.maximum(0, a)

    def gradient(self, a):
        return np.heaviside(a, 0)

class Tanh(object):
    def __call__(self,x):
        return np.tanh(x)

    def gradient(self, x):
        return 1.0 - np.tanh(x)**2

class Sigmoid(object):
    def __call__(self, a):
        output =  1 / (1 + np.exp(-a))
        self._ouptut = output
        return output

    def gradient(self, Y):
        output = self(Y)
        return output * (1 - output)


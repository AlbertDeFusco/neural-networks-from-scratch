import numpy as np

class Layer(object):
    def __init__(self, nodes, features, activation, initial_weights=None, initial_bias=None, random_state=None):
        self.activation = activation

        if random_state is not None:
            np.random.seed(random_state)

        # shape checking
        _weights = np.zeros((features, nodes))
        if initial_weights is not None:
            _weights[:] = initial_weights
        else:
            _weights[:] = np.random.randn(features, nodes) * 0.1

        _bias = np.zeros(nodes)
        if initial_bias is not None:
            _bias[:] = initial_bias

        self.bias = _bias
        self.weights = _weights

    def forward(self, X):
        self._input = X

        output = X @ self.weights + self.bias
        activated = self.activation(output)

        self._output = activated

        return activated

    def gradient(self, X, Y, G):
        act_gradient = self.activation.gradient(Y) * G

        JW = X.T.dot(act_gradient)
        Jb = np.sum(act_gradient, axis=0)
        JX = act_gradient.dot(self.weights.T)
        return JW, Jb, JX


class Network(object):
    def __init__(self, features):
        self.features = features
        self.layers = []

    def add_layer(self, nodes, activation=None, **kwargs):
        if self.layers:
            input_features = self.layers[-1].weights.shape[1]
        else:
            input_features = self.features

        if activation is None:
            activation = lambda x:s

        _layer = Layer(nodes, input_features, activation, **kwargs)
        self.layers.append(_layer)

    def predict(self, X):
        self.layer_outputs = []

        layer_input = X
        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = layer_output

            self.layer_outputs.append(layer_output)

        return self.layer_outputs[-1]

    def predict_classes(self, X):
        probabilities = self.predict(X)

        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype(np.int)

    def score(self, X, t):
        y = self.predict_classes(X).squeeze()
        return (y == t).mean()

    def train(self, X, t, loss, epochs=1000, learning_rate=0.1):
        history = []
        for epoch in range(epochs):
            y = self.predict(X).squeeze()
            output_grad = loss.gradient(y, t).reshape((-1,1))

            for layer in reversed(self.layers):
                cached_input = layer._input
                cached_output = layer._output

                dW,db,dX = layer.gradient(cached_input, cached_output, output_grad)

                output_grad = dX

                layer.weights += -learning_rate * dW
                layer.bias += -learning_rate * db

            cost = loss(y, t)
            history.append(cost)
        return history

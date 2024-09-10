import numpy as np
from utils.Optimizers import Activations, Weights_Algos, loss_compute

class layer:
    def __init__(self, input_layer : bool = False, output_layer : bool = False,
                activation : Activations = None, weigths : np.ndarray = None,
                biases : np.ndarray = None):

        self._activation = activation
        self._weights = weigths
        self._biases = biases
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._previous_layer_X = None

    def forward(self, X):
        self._previous_layer_X = X
        if self._input_layer:
            return X
        Z = np.dot(X, self._weights) + self._biases
        return self._activation.Activation_function(Z)

    def backward(self, dA, learning_rate):
        if self._input_layer:
            return dA

        X = self._previous_layer_X
        if self._output_layer:
            dZ = dA
        else:
            Z = np.dot(X, self._weights) + self._biases
            dZ = dA * self._activation.Activation_derivative(Z)
        dW = np.dot(X.T, dZ) / X.shape[0]
        db = np.sum(dZ, axis=0) / X.shape[0]

        # Update weights and biases
        self._weights -= learning_rate * dW
        self._biases -= learning_rate * db

        return np.dot(dZ, self._weights.T)


class Layers:
    AvailableActivationFunctions = ['sigmoid', 'ReLU', 'softplus', 'softmax']
    AvailableWeightsAlgos = ['heUniform']

    def __init__(self):
        self._previouslayer = 1

    def _checkArgs(self, activation, weights_initializer):
        if not isinstance(activation, str) or activation not in self.AvailableActivationFunctions:
            raise ValueError(f"'{activation}' is not a valid activation function. Available: {', '.join(self.AvailableActivationFunctions)}")

        if not isinstance(weights_initializer, str) or weights_initializer not in self.AvailableWeightsAlgos:
            raise ValueError(f"'{weights_initializer}' is not a valid weights initializer. Available: {', '.join(self.AvailableWeightsAlgos)}")

    def DenseLayer(self, shape, activation='sigmoid', weights_initializer='heUniform', input = False, output = False) -> layer:
        self._checkArgs(activation, weights_initializer)
        if activation == 'softmax' and not output:
            raise Exception("softmax is a classification activation function that could be used just in the output layer!")
        self._Activation = Activations(activation)
        if input:
            self._previouslayer = shape
            return layer(activation = self._Activation, input_layer=input, output_layer=output)

        self._Weights_initializer = Weights_Algos(weights_initializer)
        weights, self._previouslayer = self._Weights_initializer.WeightsAlgo_function((self._previouslayer, shape))
        biases = np.zeros(shape)

        return layer(activation = self._Activation, weigths = weights, biases = biases, input_layer=input, output_layer=output)

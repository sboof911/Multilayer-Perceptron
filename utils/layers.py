import numpy as np
from utils.Optimizers import Activations, Weights_Algos, loss_compute

class layer:
    def __init__(self, input : bool = None, activation : Activations = None, weigths : np.ndarray = None, biases : np.ndarray = None):
        self._activation = activation
        self._weights = weigths
        self._biases = biases
        self._input = input

    def forward(self, X):
        if self._input:
            return X
        self._Z = np.dot(X, self._weights) + self._biases

        return self._activation.Activation_function(self._Z)

    def backward(self, X, dA, learning_rate):
        dZ = dA * self._activation.Activation_derivative(self._Z)
        dW = np.dot(X.T, dZ) / X.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]

        # Update weights and biases
        gradient_desc = learning_rate * dW
        print(dZ.shape)
        print(self._weights.shape)
        self._weights -= gradient_desc
        self._biases -= learning_rate * db


class Layers:
    AvailableActivationFunctions = ['sigmoid', 'softmax', 'ReLU', 'softplus']
    AvailableWeightsAlgos = ['heUniform']

    def __init__(self, featuresNumber, outputNumber):
        self._featuresNumber = featuresNumber
        self._outputNumber = outputNumber
        self._previouslayer = 0

    def _checkArgs(self, shape, activation, weights_initializer):
        if not (isinstance(shape, int)):
            raise TypeError("Shape must be an int.")

        if not isinstance(activation, str) or activation not in self.AvailableActivationFunctions:
            raise ValueError(f"'{activation}' is not a valid activation function. Available: {', '.join(self.AvailableActivationFunctions)}")

        if not isinstance(weights_initializer, str) or weights_initializer not in self.AvailableWeightsAlgos:
            raise ValueError(f"'{weights_initializer}' is not a valid weights initializer. Available: {', '.join(self.AvailableWeightsAlgos)}")

    def DenseLayer(self, shape, activation='sigmoid', weights_initializer='heUniform') -> layer:
        self._checkArgs(shape, activation, weights_initializer)
        Activation = Activations(activation)
        if self._previouslayer == 0:
            # Need to add protection for shape and featuresNumber
            self._previouslayer = self._featuresNumber
            return layer(input = True)
        else:
            Weights_initializer = Weights_Algos(weights_initializer)
            weights, self._previouslayer = Weights_initializer.WeightsAlgo_function((self._previouslayer, shape))
            biases = np.zeros(shape)
            return layer(activation = Activation, weigths = weights, biases = biases)

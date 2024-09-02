import numpy as np
from utils.Optimizers import Activations, Weights_Algos, loss_compute

class layer:
    def __init__(self, activation : Activations, weigths : np.ndarray, biases : np.ndarray):
        self._activation = activation
        self._weights = weigths
        self._biases = biases

    def forward(self, X):
        print(f"X shape is : {X.shape}")
        print(f"weights shape is : {self._weights.shape}")
        Z = np.dot(X, self._weights) + self._biases
        print(f"Z is : {Z.shape}")
        return self._activation.Activation_function(Z)

    def backward(self, X, dA, learning_rate):
        Z = np.dot(X, self._weights) + self._biases
        dZ = dA * self._activation.Activation_derivative(Z)
        dW = np.dot(X, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = np.dot(dZ, self._weights.T)

        # Update weights and biases
        self._weights -= learning_rate * dW
        self._biases -= learning_rate * db

        return dA_prev

class Layers:
    AvailableActivationFunctions = ['sigmoid', 'softmax', 'ReLU', 'softplus']
    AvailableWeightsAlgos = ['heUniform']

    def __init__(self, featuresNumber, outputNumber):
        self._featuresNumber = featuresNumber
        self._outputNumber = outputNumber

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
        Weights_initializer = Weights_Algos(weights_initializer)
        weights = Weights_initializer.WeightsAlgo_function((self._featuresNumber, shape))
        biases = np.zeros(shape)

        return layer(Activation, weights, biases)

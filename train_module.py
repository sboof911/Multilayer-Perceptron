import pandas as pd
import numpy as np


class layers:
    AvailableActivationFunctions : list = ['sigmoid', 'softmax', 'ReLU', 'softplus']
    AvailableWeightsAlgos : list = ['heUniform']
    _shape : int
    _activation : str
    _weights_initializer : str
    _weights : np.ndarray
    _biases : np.ndarray

    def __init__(self):
        self._shape = -1
        self._activation = ""
        self._weights_initializer = ""
        self._weights = None
        self._biases = None

    def _checkArgs(self, shape,  activation, weights_initializer):
        if not isinstance(shape, [tuple, int]):
            if isinstance(shape, tuple):
                for elm in shape:
                    if not isinstance(elm, int):
                        raise Exception(f"Don t fuck with me we said ints. elm passed: {type(elm)}")
            raise Exception(f"shape must be an int or tuple of ints. shape passed: {type(shape)}")

        if not isinstance(activation, str):
            raise Exception(f"Please input the name of the activation function needed!")

        if not isinstance(weights_initializer, str):
            raise Exception(f"Please input the name of the weights_initializer algo needed!")

        if activation not in self.AvailableActivationFunctions:
            raise Exception (f"{activation} is not implemented, please select one of the available ones.\n Available Activation functions : {(' ,').join(self.AvailableActivationFunctions)}")

        if weights_initializer not in self.AvailableWeightsAlgos:
            raise Exception (f"{weights_initializer} is not implemented, please select one of the available ones.\n Available weights init algos : {(' ,').join(self.AvailableWeightsAlgos)}")

    def DenseLayer(self, shape : int,  activation : str = AvailableActivationFunctions[0], weights_initializer : str = AvailableWeightsAlgos[0]):
        self._checkArgs(shape,  activation, weights_initializer)
        self._shape = (shape) if isinstance(shape, int) else shape
        self._activation = activation
        self._weights_initializer = weights_initializer
        

class CNN_Model:
    def __init__(self):
        pass

    def createNetwork(self, layers : list):
        pass

    def fit(self, network, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.1, batch_size=8, epochs=84):
        pass

if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

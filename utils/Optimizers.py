import numpy as np

class Activations:
    def __init__(self, activation):
        self._Activation = activation

    def sigmoid_function(self, y):
        return 1 / (1 + np.exp(-y))

    def sigmoid_derivative(self, y):
        sig = self.sigmoid_function(y)
        return sig * (1 - sig)

    def softmax_function(self, y):
        exp_y = np.exp(y - np.max(y))
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def ReLU_function(self, y):
        return np.maximum(0, y)

    def ReLU_derivative(self, y):
        return np.where(y > 0, 1, 0)

    def softplus_function(self, y):
        return np.log(1 + np.exp(y))

    def softplus_derivative(self, y):
        return 1 / (1 + np.exp(-y))

    def Activation_function(self, y: np.ndarray):
        if not isinstance(y, np.ndarray):
            raise Exception("The argument must be an np.ndarray.")
        function_name = self._Activation + '_function'
        if hasattr(self, function_name):
            return getattr(self, function_name)(y)
        else:
            raise AttributeError(f"Activation function '{self._Activation}' is not defined.")

    def Activation_derivative(self, y: np.ndarray):
        function_name = self._Activation + '_derivative'
        if hasattr(self, function_name):
            return getattr(self, function_name)(y)
        else:
            raise AttributeError(f"Derivative of activation function '{self._Activation}' is not defined.")

class Weights_Algos:
    def __init__(self, WeightsAlgo):
        self._WeightsAlgo = WeightsAlgo

    def heUniform_function(self, shape):
        limit = np.sqrt(6 / shape[0])
        return np.random.uniform(-limit, limit, shape)

    def WeightsAlgo_function(self, shape):
        function_name = self._WeightsAlgo + '_function'
        if hasattr(self, function_name):
            return getattr(self, function_name)(shape)
        else:
            raise AttributeError(f"Weights initialization algorithm '{self._WeightsAlgo}' is not defined.")

class loss_compute:
    def __init__(self, loss_type):
        self._Loss_type = loss_type

    def binaryCrossentropy(self, y_true : np.ndarray, y_pred : np.ndarray):
        N = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) * (1/N)

        return loss

    def loss_function(self, y_true : np.ndarray, y_pred : np.ndarray):
        function_name = self._Loss_type
        if hasattr(self, function_name):
            return getattr(self, function_name)(y_true, y_pred)
        else:
            raise AttributeError(f"Loss type '{self._Loss_type}' not implemented.")

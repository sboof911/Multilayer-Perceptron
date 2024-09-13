import argparse
from .layers import Layers
import pandas as pd
import numpy as np

class Model_args:
    DefaultNodesNum = 24
    def __init__(self, Available_lossFunction : list, UseArg = True):
        self._featuresNum = -1
        self._outputNum = -1
        self._Available_lossFunction = Available_lossFunction
        if UseArg:
            self._args = self.Model_args()
            self._layersNodes = self._args.layer
            self._activation = self._args.activation
            self._weightalgo = self._args.weightalgo

    @property
    def featuresNum(self):
        return self._featuresNum

    @featuresNum.setter
    def featuresNum(self, newfeaturesNum):
        self._featuresNum = newfeaturesNum

    @property
    def outputNum(self):
        return self._outputNum

    @outputNum.setter
    def outputNum(self, newoutputNum):
        self._outputNum = newoutputNum

    def Check_args(self):
        if not all([True if (layer > 0 and layer < 10000) else False for layer in self._args.layer]):
            raise Exception("The minimum number of neurons in a layer is 1! and The max of layers are 10000!")

        if self._args.batch_size < 1 or self._args.batch_size > 1000:
            raise Exception("batch_size should be more then 1 and does not exceed 1000!")

        if self._args.epochs < 10 or self._args.epochs > 10000:
            raise Exception("epochs must be more then 10 and less then 10000!")

    def Model_args(self):
        parser = argparse.ArgumentParser(description="Train a neural network with specified parameters.")

        parser.add_argument("--data_train", type=str, required=True, help="Path to the datatrain file (e.g., data.csv)")
        parser.add_argument("--data_valid", type=str, default=None, help="Path to the datavalid file (e.g., data.csv)")
        parser.add_argument("--layer", type=int, nargs='+', default=[], help="List of integers representing the layers (e.g., 24 24 24)")
        parser.add_argument("--epochs", type=int, default=84, help="Number of training epochs")
        parser.add_argument("--loss", type=str, choices=self._Available_lossFunction, default=self._Available_lossFunction[0],
                            help=f"Loss function to use (e.g., {self._Available_lossFunction})")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for training")
        parser.add_argument("--weightalgo", type=str, choices=Layers.AvailableWeightsAlgos, default=Layers.AvailableWeightsAlgos[0],
                            help=f"Weights Algo for initialization to use (eg. {Layers.AvailableWeightsAlgos})")
        parser.add_argument("--activation", type=str, choices=Layers.AvailableActivationFunctions, default=Layers.AvailableActivationFunctions[0],
                            help=f"ActivationFunction to use (eg. {Layers.AvailableActivationFunctions})")

        return parser.parse_args()

    def get_DataTrain(self):
        if not self._args.data_train.endswith(".csv"):
            raise Exception("data train must be an csv file!")
        if self._args.data_valid is not None:
            if not self._args.data_valid.endswith(".csv"):
                raise Exception("data valid must be an csv file!")

        return pd.read_csv(self._args.data_train)

    def train_test_split(self, df, classes, test_size=0.2):
        y = np.array(df[classes])
        x = np.array(df.drop(classes, axis=1))
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        X_shuffled = x[indices]
        y_shuffled = y[indices]
        if self._args.data_valid is None:
            split_point = int(X_shuffled.shape[0] * (1 - test_size))
            X_train = X_shuffled[:split_point]
            y_train = y_shuffled[:split_point]
            X_valid = X_shuffled[split_point:]
            y_valid = y_shuffled[split_point:]
        else:
            X_train = X_shuffled
            y_train = y_shuffled
            df_valid = pd.read_csv(self._args.data_train)
            y_valid = np.array(df_valid[classes])
            X_valid = np.array(df_valid.drop(classes, axis=1))

        return X_train, y_train, X_valid, y_valid

    def Prepare_Layers(self):

        nodesNumList : list = self._layersNodes
        if len(nodesNumList) <= 1:
            if len(nodesNumList) == 0:
                nodesNumList.append(self.DefaultNodesNum)
            nodesNumList.append(nodesNumList[0])
        layersClass = Layers()
        layers = [layersClass.DenseLayer(self._featuresNum, activation=self._activation, input=True)]
        for nodesNum in nodesNumList:
            layers.append(layersClass.DenseLayer(nodesNum, activation=self._activation, weights_initializer=self._weightalgo))
        layers.append(layersClass.DenseLayer(self._outputNum, activation='softmax', weights_initializer=self._weightalgo, output=True))

        return layers

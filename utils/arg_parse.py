import argparse
from utils.layers import Layers
import pandas as pd

class Model_args:
    DefaultNodesNum = 24
    def __init__(self, Available_lossFunction : list, UseArg = True, featuresNum : int = -1, outputNum : int = -1):
        self._featuresNum = featuresNum
        self._outputNum = outputNum
        self._Available_lossFunction = Available_lossFunction
        if UseArg:
            self._args = self.Model_args()
            self._layersNodes = self._args.layer
            self._activation = self._args.activation
            self._weightalgo = self._args.weightalgo

    def Model_args(self):
        parser = argparse.ArgumentParser(description="Train a neural network with specified parameters.")

        parser.add_argument("--data_train", type=str, required=True, help="Path to the datatrain file (e.g., data.csv)")
        parser.add_argument("--data_valid", type=str, required=True, help="Path to the datavalid file (e.g., data.csv)")
        parser.add_argument("--layer", type=int, nargs='+', default=[], help="List of integers representing the layers (e.g., 24 24 24)")
        parser.add_argument("--epochs", type=int, default=84, help="Number of training epochs")
        parser.add_argument("--loss", type=str, choices=self._Available_lossFunction, default=self._Available_lossFunction[0],
                            help=f"Loss function to use (e.g., {self._Available_lossFunction})")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
        parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for training")
        parser.add_argument("--weightalgo", type=str, choices=Layers.AvailableWeightsAlgos, default=Layers.AvailableWeightsAlgos[0],
                            help=f"Weights Algo for initialization to use (eg. {Layers.AvailableWeightsAlgos})")
        parser.add_argument("--activation", type=str, choices=Layers.AvailableActivationFunctions, default=Layers.AvailableActivationFunctions[0],
                            help=f"ActivationFunction to use (eg. {Layers.AvailableActivationFunctions})")

        return parser.parse_args()

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
    
    def StandardizeData(self, DataFrame : pd.DataFrame, classes):
        print("Standarizing Data")
        df = DataFrame.copy()
        if classes is not None:
            df = df.drop(columns=classes)
        if self._DataFrame_std is None and self._DataFrame_mean is None:
            self._DataFrame_std = df.std()
            self._DataFrame_mean = df.mean()
        df = (df - self._DataFrame_mean) / self._DataFrame_std
        if classes is not None:
            df[classes] = DataFrame[classes]

        return df

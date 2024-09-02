import pandas as pd
import numpy as np
from utils.layers import layer, Layers, loss_compute


class CNN_Model:
    def __init__(self):
        self._Network : list = []

    def createNetwork(self, layers):
        if len(layers) - 2 < 2:
            pass
        self._Network = layers
        print(f"Network created with {len(layers)} layers.")

    def forward_propagation(self, X):
        output = X
        for layer in self._Network:
            output = layer.forward(output)
        return output

    def back_propagation(self, X, y_true, learning_rate):
        y_pred = self.forward_propagation(X)
        # Compute the initial gradient
        dA = - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))

        # Backpropagation through layers
        for layer in reversed(self._Network):
            dA = layer.backward(X, dA, learning_rate)

    def fit(self, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.1, batch_size=8, epochs=84):
        X_train, y_train = data_train
        X_valid, y_valid = data_valid
        compute_loss = loss_compute(loss)

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Train in batches
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward_propagation(X_batch)
                loss_value = compute_loss.loss_function(y_batch, y_pred, loss)
                self.back_propagation(X_batch, y_batch, learning_rate)

            # Validation loss
            y_valid_pred = self.forward_propagation(X_valid)
            valid_loss = compute_loss.loss_function(y_valid, y_valid_pred, loss)

            # Log progress
            print(f"epoch {epoch+1}/{epochs} - loss: {loss_value:.4f} - val_loss: {valid_loss:.4f}")

        print("Training complete.")

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        # Apply a threshold for binary classification, e.g., 0.5
        predicted_classes = (y_pred > 0.5).astype(int)
        return predicted_classes

if __name__ == "__main__":
    try:
        # Example usage
        X_train = np.random.rand(100, 10)  # 100 samples, 10 features
        y_train = np.random.randint(0, 4, 100)  # 100 samples, 5 output nodes for multilabel binary classification
        layers = Layers(X_train.shape[1], np.unique(y_train).size)

        model = CNN_Model()
        # model.createNetwork([layer1, layer2])
        network = model.createNetwork([
            layers.DenseLayer(X_train.shape[1], activation='sigmoid'),
            layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            layers.DenseLayer(4, activation='softmax', weights_initializer='heUniform')
            ])

        # Train the model
        model.fit((X_train, y_train), (X_train, y_train))

        # # Predict on new data
        # X_test = np.random.rand(10, 10)  # 10 new samples, 10 features
        # predictions = cnn_model.predict(X_test)
        # print("Predictions:", predictions)

    except Exception as e:
        print(f"Error: {e}")

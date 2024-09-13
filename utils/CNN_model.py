import numpy as np
from .Optimizers import loss_compute
from .plot_curve import plot_curve

class CNN_Model:
    Available_lossFunction = ["binaryCrossentropy"]

    def __init__(self) -> None:
        self._X_train_mean = None
        self._X_train_std = None
        self._class_mapping = None
        self._index_to_class_mapping = None
        self._plot_curve = plot_curve()

    def createNetwork(self, layers : list):
        self._Network = layers

    def forward_propagation(self, X):
        output = X
        for layer in self._Network:
            output = layer.forward(output)
        return output

    def back_propagation(self, X, y_true, learning_rate):
        y_pred = self.forward_propagation(X)

        # Backpropagation through layers
        dA = -2 *(y_true - y_pred)
        for layer in reversed(self._Network):
            # Compute the initial gradient
            dA = layer.backward(dA, learning_rate)

    def StandardizeData(self, X : np.ndarray):
        if self._X_train_mean is None and self._X_train_std is None:
            self._X_train_mean = X.mean(axis=0)
            self._X_train_std = X.std(axis=0)

        X = (X - self._X_train_mean) / self._X_train_std

        return X

    def map_classes_to_indices(self, data):
        if self._class_mapping is None:
            unique_classes = np.unique(data)
            self._class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
            self._index_to_class_mapping = {idx: label for label, idx in self._class_mapping.items()}

        return np.vectorize(self._class_mapping.get)(data)

    def indices_to_classes(self, data):
        if self._index_to_class_mapping is None:
            raise ValueError("Class mapping has not been initialized. Please run map_classes_to_indices first.")

        return np.vectorize(self._index_to_class_mapping.get)(data)

    def NormalizeData(self, tupple):
        x = self.StandardizeData(tupple[0])
        y = self.map_classes_to_indices(tupple[1])

        return x, y

    def fit(self, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.1, batch_size=8, epochs=84):
        X_train_data, y_train_data = self.NormalizeData(data_train)
        X_valid, y_valid = self.NormalizeData(data_valid)
        print(f"x_train shape : {X_train_data.shape}")
        print(f"x_valid shape : {X_valid.shape}")
        self._compute_loss = loss_compute(loss)

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(X_train_data.shape[0])
            np.random.shuffle(indices)
            X_train = X_train_data[indices]
            y_train = y_train_data[indices]
            y_train = np.eye(np.unique(y_train).size)[y_train]

            # Train in batches
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward_propagation(X_batch)
                loss_value = self._compute_loss.loss_function(y_batch, y_pred)
                self.back_propagation(X_batch, y_batch, learning_rate)

            # Validation loss
            y_valid_pred = np.argmax(self.forward_propagation(X_valid), axis=1)
            valid_loss = self._compute_loss.loss_function(y_valid, y_valid_pred)
            self._plot_curve.loss_array.append(valid_loss)
            self._plot_curve.acurracy_aray.append(self.accuracy(y_valid, y_valid_pred))

            # Log progress
            print(f"epoch {epoch+1}/{epochs} - loss: {loss_value:.4f} - val_loss: {valid_loss:.4f}")

        print("Training complete.")
        self._plot_curve.save()

    def predict(self, X):
        X = self.StandardizeData(X)
        y_pred = self.forward_propagation(X)
        predicted_classes = np.argmax(y_pred, axis=1)

        return self.indices_to_classes(predicted_classes)

    def accuracy(self, y_valid, y_pred, ret_loss = False):
        y_valid = np.array(y_valid)
        y_pred = np.array(y_pred)
        if y_valid.shape != y_pred.shape:
            raise ValueError("The length of y_valid and y_pred must be the same.")
        accuracy = np.mean(y_valid == y_pred) * 100
        if ret_loss == True:
            y_valid = self.map_classes_to_indices(y_valid)
            y_pred = self.map_classes_to_indices(y_pred)
            loss = self._compute_loss.loss_function(y_valid, y_pred)
            accuracy = (accuracy, loss)

        return accuracy

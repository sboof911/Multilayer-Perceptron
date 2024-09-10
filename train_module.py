import numpy as np
from utils.arg_parse import Model_args
from utils.layers import loss_compute

class CNN_Model:
    Available_lossFunction = ["binaryCrossentropy"]

    def __init__(self):
        self._Network : list = []


    def createNetwork(self, layers : list):
        self._Network = layers
        print(f"Network created with {len(layers)} layers.")

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

    def fit(self, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.1, batch_size=8, epochs=84):
        X_train_data, y_train_data = data_train
        X_valid, y_valid = data_valid
        compute_loss = loss_compute(loss)

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
                loss_value = compute_loss.loss_function(y_batch, y_pred)
                self.back_propagation(X_batch, y_batch, learning_rate)

            # Validation loss
            y_valid_pred = self.forward_propagation(X_valid)
            valid_loss = compute_loss.loss_function(y_valid, np.argmax(y_valid_pred, axis=1))

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
        y_train = np.random.randint(0, 2, 100)  # 100 samples, 5 output nodes for multilabel binary classification

        model = CNN_Model()
        parsed_elm = Model_args(CNN_Model.Available_lossFunction, featuresNum=X_train.shape[1], outputNum=np.unique(y_train).size)
        network = model.createNetwork(parsed_elm.Prepare_Layers())

        # Train the model
        model.fit((X_train, y_train), (X_train, y_train))

        # # Predict on new data
        X_test = np.random.rand(10, 10)  # 10 new samples, 10 features
        predictions = model.predict(X_test)
        print("Predictions:", predictions)

    except Exception as e:
        print(f"Error: {e}")

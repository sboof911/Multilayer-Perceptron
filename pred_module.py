import pandas as pd
import pickle, argparse
import numpy as np
from train_module import CNN_Model

def args():
    parser = argparse.ArgumentParser(description="Predict a neural network with specified parameters.")

    parser.add_argument("--data_test", type=str, required=True, help="Path to the datatest file (e.g., data.csv)")
    args = parser.parse_args()
    if not args.data_test.endswith(".csv"):
        raise Exception("data test must be an csv file!")

    return pd.read_csv(args.data_test)

classes = "class"

if __name__ == "__main__":
    try:
        DataFrame = args()
        with open('model.pkl', 'rb') as file:
            model : CNN_Model = pickle.load(file)

        prediction = model.predict(np.array(DataFrame.drop(classes, axis=1)))
        accuracy, loss = model.accuracy(np.array(DataFrame[classes]), prediction, True)
        print(prediction)
        print(f"Accuracy of the model is: {accuracy:.2f}%")
        print(f"loss : {loss:.4f}")
    except Exception as e:
        print(f"Error: {e}")

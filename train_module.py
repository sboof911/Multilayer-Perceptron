import numpy as np
from utils.arg_parse import Model_args
from utils.CNN_model import CNN_Model
import pickle, traceback

classes = "class"
if __name__ == "__main__":
    try:
        parsed_elm = Model_args(CNN_Model.Available_lossFunction)
        DataFrame = parsed_elm.get_DataTrain()
        X_train, y_train, X_valid, y_valid = parsed_elm.train_test_split(DataFrame, classes)
        parsed_elm.featuresNum = X_train.shape[1]
        parsed_elm.outputNum = np.unique(y_train).size
        model = CNN_Model()
        network = model.createNetwork(parsed_elm.Prepare_Layers())

        # Train the model
        model.fit((X_train, y_train), (X_valid, y_valid), parsed_elm._args.loss, parsed_elm._args.learning_rate,
                  parsed_elm._args.batch_size, parsed_elm._args.epochs)
        model_filename = "./model.pkl"
        print(f"> saving model {model_filename} to disk...")
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

    except Exception as e:
        print(f"Error: {e}")
        print("Traces:")
        traceback.print_exc()

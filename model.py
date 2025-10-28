import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import yaml

# Load params
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)
    
def load_model(input_shape):
    model = Sequential()
    first = True

    for layer in params["architecture"]["layers"]:
        if first:
            model.add(Dense(layer["units"], activation=layer["activation"],
                            input_shape=(input_shape)))
            first = False
        else:
            model.add(Dense(layer["units"], activation=layer["activation"]))

        if layer.get("batch_norm", False):
            model.add(BatchNormalization())

        if "dropout" in layer:
            model.add(Dropout(layer["dropout"]))

    # Output layer
    output_conf = params["architecture"]["output"]
    model.add(Dense(output_conf["units"], activation=output_conf["activation"]))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["model"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
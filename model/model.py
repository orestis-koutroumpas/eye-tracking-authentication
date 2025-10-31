import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def build_neural_net_model(input_dim, config):
    model = Sequential()
    first_layer = True
    
    for layer in config["architecture"]["layers"]:
        kwargs = dict(
            units=layer["units"],
            activation=layer["activation"]
        )
        
        if first_layer:
            kwargs["input_dim"] = input_dim
            first_layer = False

        model.add(Dense(**kwargs))

        if layer.get("batch_norm"):
            model.add(BatchNormalization())

        if layer.get("dropout"):
            model.add(Dropout(layer["dropout"]))

    output = config["architecture"]["output"]
    model.add(Dense(output["units"], activation=output["activation"]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


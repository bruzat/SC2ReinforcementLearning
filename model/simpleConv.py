from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

import model.simpleModel as simpleModel

def expand_dims(x):
    return K.expand_dims(x, 1)


class SimpleConv(simpleModel.SimpleModel):

    def __init__(self):
        super().__init__()

    def compile(self,input_dim, output_dim):
        super().compile(input_dim, output_dim)
        """Create a base network"""
        X_inputs = []
        for input_dim in dict_input_dim:
            X_inputs.append(layers.Input(shape=input_dim))

        if len(dict_input_dim) > 1:
            net = layers.Concatenate(axis=-1)(X_inputs)
        else:
            net = X_inputs[0]

        net = Lambda(expand_dims)(net)

        net = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(net)

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        softmax = layers.Activation("softmax")(net)

        self.model= Model(inputs=X_inputs, outputs=[softmax])
        self.self_value()

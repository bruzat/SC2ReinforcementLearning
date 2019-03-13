from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

import model.simpleModel as simpleModel

def expand_dims(x):
    return K.expand_dims(x, 1)


class MultiConv(simpleModel.SimpleModel):

    def __init__(self):
        super().__init__()

    def compile(self, list_input_dim, list_output_dim):
        super().compile(list_input_dim, list_output_dim)
        """Create a base network"""

        if len(list_input_dim) > 1:
            X_inputs = []
            for input_dim in list_input_dim:
                X_inputs.append(layers.Input(shape=input_dim))
            net = layers.Concatenate()(X_inputs)
        else:
            X_input = layers.Input(shape=list_input_dim[0])
            net = X_input

        net = Lambda(expand_dims)(net)

        net = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(net)

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        softmax_output = []
        for output_dim in list_output_dim:
            dense_output_dim = layers.Dense(output_dim)(net)
            softmax_output.append(layers.Activation("softmax")(dense_output_dim))

        if len(list_input_dim) > 1:
            self.model = Model(inputs=X_inputs, outputs=softmax_output)
        else:
            self.model = Model(inputs=[X_input], outputs=softmax_output)
        self.self_value()
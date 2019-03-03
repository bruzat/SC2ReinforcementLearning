from tensorflow.keras import layers
from tensorflow.keras.models import Model

import model.simpleModel as simpleModel

class SimpleDense(simpleModel.SimpleModel):

    def __init__(self):
        super().__init__()

    def compile(self,dict_input_dim, output_dim):
        super().compile(dict_input_dim, output_dim)
        """Create a base network"""
        X_inputs = []
        for input_dim in dict_input_dim:
            X_inputs.append(layers.Input(shape=input_dim))

        if len(dict_input_dim) > 1:
            net = layers.Concatenate(axis=-1)(X_inputs)
        else:
            net = X_inputs[0]

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        softmax = layers.Activation("softmax")(net)

        self.model= Model(inputs=X_inputs, outputs=[softmax])
        self.self_value()

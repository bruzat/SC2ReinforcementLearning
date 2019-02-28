from tensorflow.keras import layers
from tensorflow.keras.models import Model

import model.simpleModel as simpleModel

class SimpleDense(simpleModel.SimpleModel):

    def __init__(self):
        super().__init__()

    def compile(self,input_dim, output_dim):
        super().compile(input_dim, output_dim)
        """Create a base network"""
        X = layers.Input(shape=input_dim)
        net = X

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        softmax = layers.Activation("softmax")(net)

        self.model= Model(inputs=[X], outputs=[softmax])
        self.self_value()

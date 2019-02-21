from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

import model.simpleModel as simpleModel

def expand_dims(x):
    return K.expand_dims(x, 1)


class SimpleConv(simpleModel.SimpleModel):

    def __init__(self):
        simpleModel.SimpleModel.__init__(self)

    def compile(self,input_dim, output_dim):
        """Create a base network"""
        X = layers.Input(shape=input_dim)
        net = X

        net = Lambda(expand_dims)(net)

        net = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(net)

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)

        self.model= Model(inputs=X, outputs=net)
        self.self_value()

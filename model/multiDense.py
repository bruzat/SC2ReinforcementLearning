from tensorflow.keras import layers
from tensorflow.keras.models import Model


class MultiDense(object):

    def build_network(input_dim, output_dim):
        """Create a base network"""
        X = layers.Input(shape=input_dim)
        net = X

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)

        return Model(inputs=X, outputs=net)

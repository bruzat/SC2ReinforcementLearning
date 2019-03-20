from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

import model.baseModel as baseModel

def expand_dims(x):
    return K.expand_dims(x, 1)

class SpCMS(baseModel.BaseModel):

    def __init__(self):
        super().__init__()

    def make(self,list_input_dim, list_output_dim):
        super().make(list_input_dim, list_output_dim)
        """Create a base network"""

        X_inputs = []
        X_inputs.append(layers.Input(shape=list_input_dim[0]))
        X_inputs.append(layers.Input(shape=list_input_dim[1]))


        net1 = layers.Conv2D(64, (4, 4), padding="same", activation="relu")(X_inputs[0])
        net1 = layers.Flatten()(net1)
        net1 = layers.Dense(256)(net1)

        net2 = layers.Flatten()(X_inputs[1])
        net2 = layers.Dense(64)(net2)

        net = layers.concatenate([net1,net2])

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        softmax_output = []
        for output_dim in list_output_dim:
            dense_output_dim = layers.Dense(output_dim)(net)
            softmax_output.append(layers.Activation("softmax")(dense_output_dim))


        self.model = Model(inputs=X_inputs, outputs=softmax_output)

        self.self_value()

from tensorflow.keras import layers
from tensorflow.keras.models import Model

import model.baseModel as baseModel

class SimpleDense(baseModel.BaseModel):

    def __init__(self):
        super().__init__()

    def compile(self,list_input_dim, list_output_dim):
        super().compile(list_input_dim, list_output_dim)
        """Create a base network"""

        if len(list_input_dim) > 1:
            X_inputs = []
            X_pre_out = []
            for input_dim in list_input_dim:
                x = layers.Input(shape=input_dim)
                X_inputs.append(x)
                xt = x
                xt = layers.Flatten()(xt)
                xt = layers.Dense(256)(xt)
                X_pre_out.append(xt)
            net = layers.Concatenate()(X_pre_out)
        else:
            X_input = layers.Input(shape=list_input_dim[0])
            xt = X_input
            xt = layers.Flatten()(xt)
            xt = layers.Dense(256)(xt)
            net = xt

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

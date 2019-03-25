from tensorflow.keras import layers
from tensorflow.keras.models import Model

import model.baseModel as baseModel

class SimpleDense(baseModel.BaseModel):

    def __init__(self):
        super().__init__()

    def make(self, input_dim, output_dim, activation=None):
        super().make(input_dim=input_dim, output_dim=output_dim, activation=activation)
        """Create a base network"""

        if len(input_dim) > 1:
            X_inputs = []
            X_pre_out = []
            for inp_dim in input_dim:
                x = layers.Input(shape=inp_dim)
                X_inputs.append(x)
                xt = x
                xt = layers.Flatten()(xt)
                xt = layers.Dense(256)(xt)
                X_pre_out.append(xt)
            net = layers.Concatenate()(X_pre_out)
        else:
            X_input = layers.Input(shape=input_dim[0])
            xt = X_input
            xt = layers.Flatten()(xt)
            xt = layers.Dense(256)(xt)
            net = xt

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        softmax_output = []
        for out_dim in output_dim:
            dense_output_dim = layers.Dense(out_dim)(net)
            if self.activation != None:
                dense_output_dim=layers.Activation(self.activation)(dense_output_dim)
            softmax_output.append(dense_output_dim)

        if len(input_dim) > 1:
            self.model = Model(inputs=X_inputs, outputs=softmax_output)
        else:
            self.model = Model(inputs=[X_input], outputs=softmax_output)
        self.self_value()

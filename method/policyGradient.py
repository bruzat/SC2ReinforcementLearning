import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

from method import baseMethod

from absl import app

class PolicyGradient(baseMethod.BaseMethod):
    """
        Implementation of Policy Gradient
        This Implementation handle only continous values
    """
    def __init__(self, model, input_dim, output_dim, pi_lr, gamma, buffer_size, clipping_range, beta ):
        super().__init__(input_dim, output_dim, pi_lr, gamma, buffer_size, clipping_range, beta)

        self.model = model
        self.model.make(self.input_dim, self.output_dim, "softmax")
        self.__build_train_fn()

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        action_one_hots = []
        for i in range(len(self.output_dim)):
            action_one_hots.append(np_utils.to_categorical(act[i],self.output_dim[i]))

        result = self.model_tr.fit([*obs, adv], [*action_one_hots], epochs=5, shuffle=True, verbose=0)

        entropy = 0
        for key in result.history.keys():
            if key.endswith('f'):
                entropy += np.mean(result.history[key])

        return [np.mean(result.history['loss']),entropy,np.mean(rew)]

    def get_action(self,state):
        action = self.model.predict([[x]for x in state])

        action_prob = []
        for i in range(len(self.output_dim)):
            act = np.squeeze(action[i])
            action_prob.append(np.random.choice(np.arange(self.output_dim[i]), p=act))

        return action_prob


    def __build_train_fn(self):
        """Create a train function
        """

        # Advantages for loss function
        adv_input = Input(shape=(1,))

        self.model_tr = Model([*self.model.model.inputs, adv_input], [*self.model.model.outputs])

        adam = Adam(lr=self.pi_lr)

        self.model_tr.compile(
            optimizer=adam,
            loss=self.pg_loss(advantage=adv_input),
            metrics=[self.entropy()]
        )

    def pg_loss(self,advantage):
        def f(y_true, y_pred):
            """
            Policy gradient loss
            """
            responsible_outputs = K.sum(y_true * y_pred, axis=1)
            policy_loss = -K.mean(responsible_outputs*advantage)
            return policy_loss
        return f

    def entropy(self):
        def f(y_true, y_pred):
            return K.mean(-K.log(K.sum(y_true * y_pred, axis=1)))
        return f

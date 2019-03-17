import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils

from method import baseMethod

from absl import app
import os

class PolicyGradient(baseMethod.BaseMethod):
    """
        Implementation of Policy Gradient
        This Implementation handle only continous values
    """
    def __init__(self, model, input_dim, output_dim, pi_lr, gamma, buffer_size):
        super().__init__(input_dim, output_dim, pi_lr, gamma, buffer_size)

        self.model = model
        self.model.compile(self.input_dim, self.output_dim)
        self.__build_train_fn()

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        loss = []
        entropy = []
        for step in range(5):
            result = self.train_fn([*obs, *act, adv])
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_placeholder` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_advantage])`
        which would train the model.
        """
        action_prob_placeholder = self.model.model.outputs
        advantage_placeholder = K.placeholder(shape=(None,),
                                                    name="advantage")
        action_placeholder = []
        action_prob = []
        loss = []
        for i in range(len(self.output_dim)):
            act_pl = K.placeholder(shape=(None,),
                                   name="action_placeholder"+str(i),
                                   dtype='int32')
            action_placeholder.append(act_pl)

            act_prob = K.sum(K.one_hot(act_pl,self.output_dim[i])
                                        * action_prob_placeholder[i] , axis=1)
            act_prob = K.log(act_prob)
            action_prob.append(K.mean(-act_prob))

            l = -K.mean(act_prob * advantage_placeholder)
            loss.append(l)

        entropy = K.sum(action_prob)
        loss = K.stack(loss)
        loss_p = K.sum(loss)


        adam = optimizers.Adam(lr = self.pi_lr)
        updates=adam.get_updates(loss=loss,
                                        params=self.model.trainable_weights)

        self.train_fn = K.function(inputs=[*self.model.model.inputs,
                                           *action_placeholder,
                                           advantage_placeholder],
                                    outputs=[loss_p,entropy],updates=updates)

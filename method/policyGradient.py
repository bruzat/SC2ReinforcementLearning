import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils

from method import simpleMethod

from absl import app
import os

class PolicyGradient(simpleMethod.SimpleMethod):
    """
        Implementation of Proximal Policy Optimization
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
            result = self.train_fn([obs, act, adv])
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
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output[0]
        action_placeholder = K.placeholder(shape=(None,),
                                                  name="action_placeholder",
                                                  dtype='int32')
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * action_prob_placeholder , axis=1)
        action_prob = K.log(action_prob)

        loss = action_prob * reward_placeholder
        loss = -K.mean(loss)

        entropy = K.mean(-action_prob)

        adam = optimizers.Adam(lr = self.pi_lr)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_placeholder,
                                           reward_placeholder],
                                   outputs=[loss,entropy],updates=updates)

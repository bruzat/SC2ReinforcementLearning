import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras import losses

from method import simpleMethod
import os

class TrustRegionPolicyOptomisation(simpleMethod.SimpleMethod):
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
        old_mu = self.get_actions_values(obs)
        for step in range(5):
            result = self.train_fn([obs, old_mu, act, adv])
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def get_actions_values(self, states):
        return np.squeeze(self.model.predict(states))

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.

        """
        action_prob_placeholder = self.model.output
        old_mu_placeholder = K.placeholder(shape=(None,),
                                                  name="old_mu_placeholder")
        action_placeholder = K.placeholder(shape=(None,),
                                                  name="action_placeholder",
                                                  dtype='int32')
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="reward")


        action_prob = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * action_prob_placeholder , axis=1)

        action_prob_old = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * old_mu_placeholder , axis=1)

        logp = K.log(action_prob)
        old_logp = K.log(action_prob_old)
        kl  = losses.kullback_leibler_divergence(old_mu_placeholder,action_prob_placeholder)


        loss = (action_prob-action_prob_old) * reward_placeholder - kl
        loss = -K.mean(loss)

        entropy = K.mean(-K.log(action_prob_old))
        adam = optimizers.Adam(lr = self.pi_lr)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           old_mu_placeholder,
                                           action_placeholder,
                                           reward_placeholder],
                                   outputs=[loss,entropy],updates=updates)

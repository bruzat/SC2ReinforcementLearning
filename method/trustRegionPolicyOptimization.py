import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras import losses

from method import baseMethod
import os

class TrustRegionPolicyOptimization(baseMethod.BaseMethod):
    """
        Implementation of Trust Region Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, model, input_dim, output_dim, pi_lr, gamma, buffer_size):
        super().__init__(input_dim, output_dim, pi_lr, gamma, buffer_size)

        self.model = model
        self.model.make(self.input_dim, self.output_dim)
        self.__build_train_fn()

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        loss = []
        entropy = []
        old_mu = self.get_actions_values(obs)
        for step in range(5):
            result = self.train_fn([*obs, *old_mu, *act, adv])
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        """
        action_prob_placeholder = self.model.model.outputs
        advantage_placeholder = K.placeholder(shape=(None,),
                                                    name="advantage")

        action_placeholder = []
        old_mu_placeholder = []
        action_prob_old = []
        loss = []
        for i in range(len(self.output_dim)):
            o_mu_pl = K.placeholder(shape=(None,),
                                    name="old_mu_placeholder"+str(i))
            old_mu_placeholder.append(o_mu_pl)

            act_pl = K.placeholder(shape=(None,),
                                   name="action_placeholder"+str(i),
                                   dtype='int32')
            action_placeholder.append(act_pl)

            act_prob = K.sum(K.one_hot(act_pl,self.output_dim[i])
                                        * action_prob_placeholder[i] , axis=1)

            act_prob_old = K.sum(K.one_hot(act_pl,self.output_dim[i])
                                        * o_mu_pl , axis=1)
            action_prob_old.append(K.mean(-K.log(act_prob_old)))

            logp = K.log(act_prob)
            old_logp = K.log(act_prob_old)
            kl  = losses.kullback_leibler_divergence(old_mu_placeholder[i],action_prob_placeholder[i])

            l = (act_prob-act_prob_old) * advantage_placeholder - kl
            loss.append(-K.mean(l))

        entropy = K.sum(action_prob_old)
        loss = K.stack(loss)
        loss_p = K.sum(loss)

        adam = optimizers.Adam(lr = self.pi_lr)
        updates=adam.get_updates(loss=loss,
                                        params=self.model.trainable_weights)

        self.train_fn = K.function(inputs=[*self.model.model.inputs,
                                           *old_mu_placeholder,
                                           *action_placeholder,
                                           advantage_placeholder],
                                    outputs=[loss_p,entropy],updates=updates)

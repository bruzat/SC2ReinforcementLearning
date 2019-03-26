import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras import losses

from tensorflow.keras.optimizers import Adam

from method import baseMethod

class ProximalPolicyOptimization(baseMethod.BaseMethod):
    """
        Implementation of Proximal Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, model, input_dim, output_dim, pi_lr, gamma, buffer_size, clipping_range, beta ):
        super().__init__(input_dim, output_dim, pi_lr, gamma, buffer_size, clipping_range, beta)

        self.model = model
        self.model.make(self.input_dim, self.output_dim, "softmax")
        self.critic = model.__class__()
        self.critic.make(self.input_dim, [1], None)
        self.critic.compile(optimizer=Adam(lr=self.pi_lr), loss='mse')

        self.__build_train_fn()

    def save(self,path, name):
        super().save(path, name)
        self.critic.save_model(path)

    def load(self, path, name):
        super().load(path, name)
        self.critic.load_model(path+'critic/'+name)
        self.critic.compile(optimizer=Adam(lr=self.pi_lr), loss='mse')
        return int(it)

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        loss = []
        entropy = []
        old_mu = self.get_actions_values(obs)
        pred_values = self.critic_predict(obs)
        adv_new = np.subtract(adv,pred_values)

        for step in range(5):
            result = self.train_fn([ *obs, *old_mu, *act, adv_new])
            self.critic.model.fit(obs, [pred_values], verbose=0)
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def critic_predict(self, obs):
        pred = self.critic.predict(obs)
        return [i[0] for i in pred]


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

            r = act_prob/(act_prob_old + 1e-10)

            l = K.minimum(r * advantage_placeholder, K.clip(r, min_value=1-self.clipping_range, max_value=1+self.clipping_range) * advantage_placeholder)
            l = l + self.beta * (act_prob * K.log(act_prob + 1e-10))
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

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from method import baseMethod

from absl import app

class ProximalPolicyOptimization(baseMethod.BaseMethod):
    """
        Implementation of Policy Gradient
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
        action_one_hots = []
        for i in range(len(self.output_dim)):
            action_one_hots.append(np_utils.to_categorical(act[i],self.output_dim[i]))
        old_mu = self.get_actions_values(obs)
        pred_values = self.critic_predict(obs)
        adv_new = np.subtract(adv,pred_values)

        result = self.model_tr.fit([*obs, *old_mu, adv_new],[*action_one_hots], epochs=5, shuffle=True, verbose=0)
        self.critic.model.fit([*obs], [pred_values], epochs=5, shuffle=True, verbose=0)

        entropy = 0
        for key in result.history.keys():
            if key.endswith('f'):
                entropy += np.mean(result.history[key])

        return [np.mean(result.history['loss']),entropy,np.mean(rew)]

    def critic_predict(self, obs):
        pred = self.critic.predict(obs)
        return [i[0] for i in pred]


    def __build_train_fn(self):
        """Create a train function
        """

        old_prediction = []
        for shape in self.output_dim:
            old_prediction.append(Input(shape=(shape,)))

        # Advantages for loss function
        adv_input = Input(shape=(1,))

        self.model_tr = Model([*self.model.model.inputs, *old_prediction, adv_input], [*self.model.model.outputs])

        adam = Adam(lr=self.pi_lr)

        self.model_tr.compile(
            optimizer=adam,
            loss=[ self.proximal_policy_optimization_loss(advantage=adv_input,old_prediction=old_prediction[i]) for i in range(len(self.output_dim))],
            metrics=[ProximalPolicyOptimization.entropy()]
        )

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage,
                                    K.clip(r, min_value=1 - self.clipping_range, max_value=1 + self.clipping_range) * advantage) + self.beta * (prob * K.log(prob + 1e-10)))
        return loss

    def entropy():
        def f(y_true, y_pred):
            return K.mean(-K.log(K.sum(y_true * y_pred, axis=1)))
        return f

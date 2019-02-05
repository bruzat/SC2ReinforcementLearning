import numpy as np
import tensorflow as tf
import scipy
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import Activation


def log_softmax(x):
    return x - K.log(K.sum(K.exp(x)))

np_utils.get_custom_objects().update({'log_softmax': Activation(log_softmax)})

def multinomial_log(N, logp):
    log_rand = -np.random.exponential(size=N)
    logp_cuml = np.logaddexp.accumulate(np.hstack([[-np.inf], logp]))
    logp_cuml -= logp_cuml[-1]
    return np.histogram(log_rand, bins=logp_cuml)[0]


class Buffer:
    """
    A buffer for storing trajectories experienced by a PolicyGradient agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma):
        self.size = size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.obs_buf = np.zeros(Buffer.combined_shape(self.size, self.obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(self.size, dtype=np.int32)
        # Rewards buffer
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.path_start_idx= 0, 0

    def store(self, obs, act, rew):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def get(self):
        self.ptr = 0
        self.path_start_idx = 0
        obs = self.obs_buf
        act = self.act_buf
        rew = self.rew_buf
        adv = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        self.obs_buf = np.zeros(Buffer.combined_shape(self.size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.size, dtype=np.int32)
        self.rew_bu = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        return obs, act, rew, adv

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Advantage
        self.adv_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    @staticmethod
    def discount_cumsum(x, discount):
        """
            x = [x0, x1, x2]
            output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PolicyGradient(object):
    """
        Implementation of Proximal Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, input_dim, output_dim, pi_lr, gamma, buffer_size):
        super(PolicyGradient, self).__init__()
        # Stored the spaces
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.buffer = Buffer(self.input_dim,self.output_dim,buffer_size,gamma)
        # Learning rate of the policy network
        self.pi_lr = pi_lr

        self.__build_network(self.input_dim, self.output_dim)
        self.__build_train_fn()

    def store(self, obs, act, rew):
        self.buffer.store(obs, act, rew)
        return

    def save(self, it):
        self.model.save('./logger/model_'+str(it)+'.h5')


    def load(self, model):
        saves = [int(x[11:]) for x in os.listdir("./logger") if model in x and len(x) > 11]
        it = '%d' % max(saves)
        self.model = Model.load_model('./logger/'+str(model)+'_'+str(it)+'.h5')
        self.__build_train_fn()
        return it

    def finish_path(self, last_val):
        self.buffer.finish_path(last_val)

    def get_action(self, state):
        pred = self.model.predict(state)[0]
        multi = multinomial_log(1,pred)

        return multi.argmax()

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        loss = []
        entropy = []
        for step in range(5):
            result = self.train_fn([obs, act, adv])
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def __build_network(self, input_dim, output_dim):
        """Create a base network"""
        self.X = layers.Input(shape=input_dim)
        net = self.X

        net = layers.Flatten()(net)

        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)

        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output
        action_placeholder = K.placeholder(shape=(None,),
                                                  name="action_placeholder",
                                                  dtype='int32')
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob_placeholder_p = K.print_tensor(action_prob_placeholder)
        action_prob_logsoftmax = log_softmax(action_prob_placeholder_p)
        action_prob_logsoftmax = K.print_tensor(action_prob_logsoftmax)

        action_prob = K.sum(K.one_hot(action_placeholder,self.output_dim) * action_prob_logsoftmax , axis=1)

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

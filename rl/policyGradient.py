import numpy as np
from scipy import signal
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import Activation, Lambda

from absl import app

def log_softmax(x):
    return x - K.log(K.sum(K.exp(x)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))

np_utils.get_custom_objects().update({'log_softmax': Activation(log_softmax)})

class Buffer:
    """
    A buffer for storing trajectories experienced by a PolicyGradient agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(Buffer.combined_shape(size, obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    @staticmethod
    def discount_cumsum(x, discount):
        """
            x = [x0, x1, x2]
            output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Advantage
        self.adv_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # Normalize the Advantagee
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return self.obs_buf, self.act_buf, self.rew_buf, self.adv_buf

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

    def save(self, it):
        self.model.save('./logger/model_'+str(it)+'.h5')


    def load(self, model):
        saves = [int(x[11:]) for x in os.listdir("./logger") if model in x and len(x) > 11]
        it = '%d' % max(saves)
        self.model = Model.load_model('./logger/'+str(model)+'_'+str(it)+'.h5')
        self.__build_train_fn()
        return it

    def finish_path(self,last_val):
        self.buffer.finish_path(last_val)

    def get_action(self, state):
        action_prob = np.squeeze(self.model.predict([state]))
        return np.random.choice(np.arange(self.output_dim), p=action_prob)


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
        net = layers.Activation("softmax")(net)

        self.model = Model(inputs=self.X, outputs=net)

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
        action_prob_placeholder = self.model.output
        action_placeholder = K.placeholder(shape=(None,),
                                                  name="action_placeholder",
                                                  dtype='int32')
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob_logsoftmax = K.log(action_prob_placeholder)

        action_prob = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * action_prob_logsoftmax , axis=1)

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

class GridWorld(object):
    """
        docstring for GridWorld.
    """
    def __init__(self):
        super(GridWorld, self).__init__()

        self.rewards = [
            [0,  0,  0, 0, -1, 0, 0],
            [0, -1, -1, 0, -1, 0, 0],
            [0, -1, -1, 1, -1, 0, 0],
            [0, -1, -1, 0, -1, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
        ]
        self.position = [6, 6] # y, x

    def gen_state(self):
        # Generate a state given the current position of the agent
        state = np.zeros((7, 7))
        state[self.position[0]][self.position[1]] = 1
        return state

    def step(self, action):
        if action == 0: # Top
            self.position = [(self.position[0] - 1) % 7, self.position[1]]
        elif action == 1: # Left
            self.position = [self.position[0], (self.position[1] - 1) % 7]
        elif action == 2: # Right
            self.position = [self.position[0], (self.position[1] + 1) % 7]
        elif action == 3: # Down
            self.position = [(self.position[0] + 1) % 7, self.position[1]]

        reward = self.rewards[self.position[0]][self.position[1]]
        done = False if reward == 0 else True
        state = self.gen_state()
        if done: # The agent is dead, reset the game
            self.position = [6, 6]
        return state, reward, done

    def display(self):
        y = 0
        print("="*14)
        for line in self.rewards:
            x = 0
            for case in line:
                if case == -1:
                    c = "0"
                elif (y == self.position[0] and x == self.position[1]):
                    c = "A"
                elif case == 1:
                    c = "T"
                else:
                    c = "-"
                print(c, end=" ")
                x += 1
            y += 1
            print()

def main(_):
    grid = GridWorld()
    buffer_size = 1000

    # Create the NET class
    agent = PolicyGradient(
    	input_dim=(7, 7),
    	output_dim=4,
    	pi_lr=0.001,
    	buffer_size=buffer_size,
        gamma=0.99
    )

    rewards = []

    b = 0

    for epoch in range(10000):

        done = False
        state = grid.gen_state()

        while not done:
            action = agent.get_action([state])
            n_state, reward, done = grid.step(action)
            agent.store(state, action, reward)
            b += 1

            state = n_state

            if done:
                agent.finish_path(reward)
                rewards.append(reward)
                if len(rewards) > 1000:
                    rewards.pop(0)
            if b == buffer_size:
                if not done:
                    agent.finish_path(0)
                agent.train()
                b = 0

        if epoch % 1000 == 0:
            print("Rewards mean:%s" % np.mean(rewards))

    for epoch in range(10):
        import time
        print("=========================TEST=================================")
        done = False
        state = grid.gen_state()

        while not done:
            time.sleep(1)
            action = agent.get_action([state])
            n_state, reward, done = grid.step(action)
            grid.display()
            state = n_state
        print("reward=>", reward)

if __name__ == '__main__':
    app.run(main)

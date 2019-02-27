import numpy as np
from scipy import signal

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


class SimpleMethod(object):
    def __init__(self, input_dim, output_dim, pi_lr, gamma, buffer_size):
        super().__init__()
        # Stored the spaces
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.buffer = Buffer(self.input_dim,self.output_dim,buffer_size,gamma)
        # Learning rate of the policy network
        self.pi_lr = pi_lr
        self.model = None

    def save(self,path,method,model, it):
        writepath='./'+path+'/'+method+'/'+model+'/'+model+str(it)+'.h5'
        self.model.save(writepath)

    def load(self,path,method, model):
        saves = [int(x[len(model):-3]) for x in os.listdir('./'+path+'/'+method+'/'+model) if model in x and len(x) > len(model)]
        it = '%d' % max(saves)
        writepath='./'+path+'/'+method+'/'+str(model)+'/'+str(model)+str(it)+'.h5'
        self.model.load(writepath)
        self.__build_train_fn()
        return int(it)

    def store(self, obs, act, rew):
        self.buffer.store(obs, act, rew)

    def finish_path(self,last_val):
        self.buffer.finish_path(last_val)

    def get_action(self, state):
        action_prob = np.squeeze(self.model.predict([state])[0])
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils as np_utils

from method import simpleMethod

from absl import app
import os

class TrustRegionPolicyOptomisation(object):
    """
        Implementation of Proximal Policy Optimization
        This Implementation handle only continous values
    """
    def __init__(self, model, input_dim, output_dim, pi_lr, gamma, buffer_size):
        super().__init__(input_dim, output_dim, pi_lr, gamma, buffer_size)

        self.model = model
        self.model.compile(self.input_dim, self.output_dim)
        self.old_model = model.duplicate_model()
        self.__build_train_fn()

    def train(self):
        obs, act, rew, adv = self.buffer.get()
        loss = []
        entropy = []
        for step in range(5):
            result = self.train_fn([obs, obs, act, adv])
            loss.append(result[0])
            entropy.append(result[1])

        return [np.mean(loss),np.mean(entropy),np.mean(rew)]

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        """
        action_prob_placeholder = self.model.output
        action_prob_placeholder_old = self.old_model.output
        action_placeholder = K.placeholder(shape=(None,),
                                                  name="action_placeholder",
                                                  dtype='int32')
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="reward")

        action_prob = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * action_prob_placeholder , axis=1)

        action_prob_old = K.sum(K.one_hot(action_placeholder,self.output_dim)
                                    * action_prob_placeholder_old , axis=1)

        loss = (action_prob/action_prob_old) * reward_placeholder
        loss = -K.mean(loss)

        entropy = K.mean(-K.log(action_prob))

        adam = optimizers.Adam(lr = self.pi_lr)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           self.old_model.input,
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

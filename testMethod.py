from method import trustRegionPolicyOptimization, policyGradient, proximalPolicyOptimization
from model import simpleDense
from absl import app

import numpy as np

class GridWorld(object):
    """
        docstring for GridWorld.
    """
    def __init__(self, dim = 2):
        super(GridWorld, self).__init__()
        self.dim = dim

        self.rewards = []
        self.position = []
        for i in range(self.dim):
            self.rewards.append([
                [0,  0,  0, 0, -1, 0, 0],
                [0, -1, -1, 0, -1, 0, 0],
                [0, -1, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, -1, 0, 0],
                [0,  0,  0, 0,  0, 0, 0],
                [0,  0,  0, 0,  0, 0, 0],
                [0,  0,  0, 0,  0, 0, 0],
            ])
            self.position.append([6, 6]) # y, x

    def gen_state(self):
        # Generate a state given the current position of the agent
        state = []
        for i in range(self.dim):
            s = np.zeros((7, 7))
            s[self.position[i][0]][self.position[i][1]] = 1
            state.append(s)
        return state

    def step(self, action):
        state = []
        reward = []
        done = []
        for i in range(self.dim):
            if action[i] == 0: # Top
                self.position[i] = [(self.position[i][0] - 1) % 7, self.position[i][1]]
            elif action[i] == 1: # Left
                self.position[i] = [self.position[i][0], (self.position[i][1] - 1) % 7]
            elif action[i] == 2: # Right
                self.position[i] = [self.position[i][0], (self.position[i][1] + 1) % 7]
            elif action[i] == 3: # Down
                self.position[i] = [(self.position[i][0] + 1) % 7, self.position[i][1]]
            reward.append(self.rewards[i] [self.position[i][0]] [self.position[i][1]])
            done.append(False) if reward[i] == 0 else done.append(True)
        dones = True
        for e in done:
            if e != True:
                dones = False
        if dones: # The agent is dead, reset the game
            for i in range(self.dim):
                self.position[i] = [6, 6]
        state = self.gen_state()
        reward = np.sum(reward)


        return state, reward, dones

    def display(self):

        print("="*14)
        for i in range(self.dim):
            y = 0
            print('.'*10)
            print(self.position[i])
            for line in self.rewards[i]:
                x = 0
                for case in line:
                    if case == -1:
                        c = "0"
                    elif (y == self.position[i][0] and x == self.position[i][1]):
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
    grid = GridWorld(dim = 1)
    buffer_size = 1000

    # Create the NET class
    agent = proximalPolicyOptimization.ProximalPolicyOptimization(
    	input_dim=[(7, 7)],
    	output_dim=[4],
    	pi_lr=0.001,
    	buffer_size=buffer_size,
        gamma=0.99,
        model=simpleDense.SimpleDense()
    )

    rewards = []

    b = 0

    for epoch in range(10000):

        done = False
        state = grid.gen_state()

        while not done:
            action = agent.get_action(state)
            n_state, reward, done = grid.step(action)
            agent.store(state, action, reward)
            b += 1

            state = n_state

            if done:
                agent.finish_path(reward)
                if len(rewards) > 100000:
                    print("pop")
                    for i in range(1000):
                        rewards.pop(0)
                rewards.append(reward)
            if b >= buffer_size:
                if not done:
                    agent.finish_path(0)
                    done = True

                agent.train()
                b = 0

        if epoch % 1000 == 0:
            print("Rewards mean:%s" % np.mean(rewards))
        if epoch % 100 == 0:
            print(1/10 * epoch / 100)


    for epoch in range(10):
        import time
        print("=========================TEST=================================")
        done = False
        state = grid.gen_state()

        while not done:
            time.sleep(1)
            action = agent.get_action(state)
            n_state, reward, done = grid.step(action)
            grid.display()
            state = n_state
        print("reward=>", reward)

if __name__ == '__main__':
    app.run(main)

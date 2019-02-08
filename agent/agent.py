from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np

class Agent(base_agent.BaseAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self,  path='logger', model='model', load_model=False, rl=None):
		super(Agent, self).__init__()
		self.model = model
		self.nb_steps = 0
		self.max_steps = 512
		self.epoch = 0
		self.path = path

        # Create the NET class
		self.rl = rl(
        	input_dim=(64,64),
        	output_dim=64*64,
        	pi_lr=0.0001,
        	gamma=0.99,
        	buffer_size=512,
		)


		# Load the model
		if load_model:
            #Load the existing model
			self.epoch = self.rl.load(self.path,self.model)


	def train(self, obs_new, obs, action, reward):
		# Train the agent
		reward = -1 if reward == 0 else 1
		feat = Agent.get_feature_screen(obs, features.SCREEN_FEATURES.player_relative)
		# Store the reward
		action_r = action[0]*64 + action[1]
		self.rl.store(feat, action_r, reward)
		# Increase the current step
		self.nb_steps += 1
		# Finish the episode on reward == 1
		if reward == 1 and self.nb_steps != self.max_steps and not obs_new.last():
			self.rl.finish_path(reward)
		# If this is the end of the epoch or this is the last observation
		if self.nb_steps == self.max_steps or obs_new.last():
			# If this is the last observation, we bootstrap the value function
			self.rl.finish_path(reward)

			# We do not train yet if this is just the end of thvvve current episode
			if obs_new.last() is True and self.nb_steps != self.max_steps:
				return

			result = self.rl.train()
			self.rl.print_train_result(self.epoch, result)
			self.rl.log_train_result(self.path,self.model,self.epoch, result)


			self.nb_steps = 0
			self.epoch += 1
			# Save every 100 epochs
			if (self.epoch-1) % 300 == 0:
				self.rl.save(self.path,self.model,self.epoch)

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(Agent, self).step(obs)
		# if we can move our army (we have something selected)
		if actions.FUNCTIONS.Move_screen.id in obs.observation['available_actions']:
			# Get the features of the screen
			feat = Agent.get_feature_screen(obs, features.SCREEN_FEATURES.player_relative)
        	# Step with ppo according to this state
			act = self.rl.get_action([feat])
			# Convert the prediction into positions
			positions = Agent.prediction_to_position([act])
			# Get a random location on the map
			return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], positions[0]])

		# if we can't move, we havent selected our army, so selecto ur army
		else:
			return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])

	@staticmethod
	def get_feature_screen(obs, screen_feature):
		# Get the feature associated with the observation
		mapp = obs.observation["feature_screen"][screen_feature.index]
		return np.array(mapp)

	@staticmethod
	def prediction_to_position(pi, dim = 64):
		# Translate the prediction to y,x position
		pirescale = np.expand_dims(pi, axis=1)
		pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
		positions = np.zeros_like(pirescale)
		positions[:,0] = pirescale[:,0] // dim
		positions[:,1] = pirescale[:,0] % dim
		return positions

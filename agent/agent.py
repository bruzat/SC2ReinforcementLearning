from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np

import agent.log as log

class Agent(base_agent.BaseAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self,  path='logger/', model_name='model', model=None, load_model=False, method_name="method", method=None):
		super(Agent, self).__init__()
		self.logger = log.Logger()
		self.model_name = model_name
		self.method_name = method_name
		self.nb_steps = 0
		self.max_steps = 512
		self.epoch = 0
		self.path = path

        # Create the NET class
		self.method = method(
			model = model,
        	input_dim=(64,64),
        	output_dim=64*64,
        	pi_lr=0.001,
        	gamma=0.98,
        	buffer_size=512,
		)


		# Load the model
		if load_model:
            #Load the existing model
			self.epoch = self.method.load(self.path, self.method, self.model_name)

	def train(self, obs_new, obs, action, reward):
		# Train the agent
		reward = 0 if reward == 0 else 1
		feat = Agent.get_feature_screen(obs, features.SCREEN_FEATURES.player_relative)
		# Store the reward
		action_r = action[0]*64 + action[1]
		self.method.store(feat, action_r, reward)
		# Increase the current step
		self.nb_steps += 1
		# Finish the episode on reward == 1
		if reward == 1 and self.nb_steps != self.max_steps and not obs_new.last():
			self.method.finish_path(reward)
		# If this is the end of the epoch or this is the last observation
		if self.nb_steps == self.max_steps or obs_new.last():
			# If this is the last observation, we bootstrap the value function
			self.method.finish_path(-1)

			# We do not train yet if this is just the end of thvvve current episode
			if obs_new.last() is True and self.nb_steps != self.max_steps:
				return

			result = self.method.train()
			self.logger.print_train_result(self.epoch, result)
			self.logger.log_train_result(self.path, self.method_name, self.model_name, self.epoch, result)


			self.nb_steps = 0
			self.epoch += 1
			# Save every 100 epochs
			if (self.epoch-1) % 300 == 0:
				self.method.save(self.path,self.method_name,self.model_name,self.epoch)

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(Agent, self).step(obs)
		# if we can move our army (we have something selected)
		if actions.FUNCTIONS.Move_screen.id in obs.observation['available_actions']:
			# Get the features of the screen
			feat = Agent.get_feature_screen(obs, features.SCREEN_FEATURES.player_relative)
        	# Step with ppo according to this state
			act = self.method.get_action([feat])
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

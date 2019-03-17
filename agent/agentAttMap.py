from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np

import agent.log as log

class AgentAttMap(base_agent.BaseAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self,  path='logger/', model_name='model', model=None, load_model=False, method_name="method", method=None):
		super(AgentFindAndDefeatZerglings, self).__init__()
		self.logger = log.Logger()
		self.model_name = model_name
		self.method_name = method_name
		self.nb_steps = 0
		self.max_steps = 2056
		self.epoch = 0
		self.path = path
		self.score = 0
		self.score_reset = 0

        # Create the NET class
		self.method = method(
			model = model,
        	input_dim=[(5,64,64),(4,64,64),(3,7)],
        	output_dim=[3,64*64,64*64],
        	pi_lr=0.00001,
        	gamma=0.98,
        	buffer_size=2056
		)


		# Load the model
		if load_model:
            #Load the existing model
			self.epoch = self.method.load(self.path, self.method_name, self.model_name)

		self.logger.drawModel(self.method.model.model, self.path, self.method_name, self.model_name)

	def train(self, obs_new, obs, action, reward):
		# Train the agent
		self.score += reward
		if reward == -1:
			reward = -50
		elif reward == 0:
			reward = -1

		feat = AgentFindAndDefeatZerglings.get_feature_screen(obs)
		# Store the reward
		self.method.store(feat, action, reward)
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
			if obs_new.last():
				self.score_reset += 1
			if obs_new.last() is True and self.nb_steps != self.max_steps:
				return

			result = self.method.train()
			self.logger.print_train_result(self.epoch, result, self.score//self.score_reset)
			self.logger.log_train_result(self.path, self.method_name, self.model_name, self.epoch, self.score//self.score_reset, result)

			self.score_reset = 0
			self.score = 0
			self.nb_steps = 0
			self.epoch += 1
			# Save every 100 epochs
			if (self.epoch-1) % 50 == 0:
				self.method.save(self.path,self.method_name,self.model_name,self.epoch)

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(AgentFindAndDefeatZerglings, self).step(obs)
		# if we can move our army (we have something selected)
		# Get the features of the screen
		feat = AgentFindAndDefeatZerglings.get_feature_screen(obs)
    	# Step with ppo according to this state
		act = self.method.get_action(feat)

		if act[0] == 0:
			if actions.FUNCTIONS.Move_screen.id in obs.observation['available_actions']:
				# Convert the prediction into positions
				position = AgentFindAndDefeatZerglings.prediction_to_position([act[1]])
				# Get a random location on the map
				return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], position[0]]) , act
			else:
				return actions.FunctionCall(actions.FUNCTIONS.no_op.id,[]), act
		elif act[0] == 1:
			if actions.FUNCTIONS.Attack_screen.id in obs.observation['available_actions']:
				# Convert the prediction into positions
				position = AgentFindAndDefeatZerglings.prediction_to_position([act[1]])
				# Get a random location on the map
				return actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0], position[0]]) , act
			else:
				return actions.FunctionCall(actions.FUNCTIONS.no_op.id,[]), act
		elif act[0] == 2:
			if actions.FUNCTIONS.move_camera.id in obs.observation['available_actions']:
				# Convert the prediction into positions
				position = AgentFindAndDefeatZerglings.prediction_to_position([act[2]])
				# Get a random location on the map
				return actions.FunctionCall(actions.FUNCTIONS.move_camera.id, [position[0]]) , act
			else:
				return actions.FunctionCall(actions.FUNCTIONS.no_op.id,[]), act


	@staticmethod
	def get_feature_screen(obs):
		# Get the feature associated with the observation
		mapp = []
		mapp.append(obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index])
		mapp.append(obs.observation.feature_screen[features.SCREEN_FEATURES.visibility_map.index])
		mapp.append(obs.observation.feature_screen[features.SCREEN_FEATURES.unit_hit_points.index])
		mapp.append(obs.observation.feature_screen[features.SCREEN_FEATURES.selected.index])
		mapp.append(obs.observation.feature_screen[features.SCREEN_FEATURES.unit_density.index])

		minimapp = []
		minimapp.append(obs.observation.feature_minimap[features.MINIMAP_FEATURES.visibility_map.index])
		minimapp.append(obs.observation.feature_minimap[features.MINIMAP_FEATURES.camera.index])
		minimapp.append(obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index])
		minimapp.append(obs.observation.feature_minimap[features.MINIMAP_FEATURES.selected.index])

		multi_select = np.zeros((3,7))
		obs_mult = obs.observation["multi_select"]
		for i in range(len(multi_select)):
			if len(obs_mult) > i:
				multi_select[i] = obs_mult[i]
			else:
				break
		return [np.array(mapp),np.array(minimapp),multi_select]

	@staticmethod
	def prediction_to_position(pi, dim = 64):
		# Translate the prediction to y,x position
		pirescale = np.expand_dims(pi, axis=1)
		pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
		positions = np.zeros_like(pirescale)
		positions[:,0] = pirescale[:,0] // dim
		positions[:,1] = pirescale[:,0] % dim
		return positions
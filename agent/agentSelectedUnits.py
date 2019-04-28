from agent import baseAgent
from pysc2.lib import actions, features
import numpy as np
import os

import agent.log as log

class AgentSelectedUnits(baseAgent.BaseAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self, model, path='logger/', model_name='model', method_name="method", method=None, load_model=False,
 					val_null=0, coef_neg=1, coef_pos=1, pi_lr=0.001,gamma=0.98, buffer_size=1024, clipping_range=0.2,
					beta=1e-3, map_size = 80, minimap_size = 64):
		super().__init__(model, path=path, model_name=model_name, method_name=method_name, method=method, load_model=load_model,
		 					val_null=val_null, coef_neg=coef_neg, coef_pos=coef_pos, pi_lr=pi_lr, gamma=gamma, buffer_size=buffer_size,
							clipping_range=clipping_range, beta=beta, map_size = map_size, minimap_size = minimap_size)        # Create the NET class
		self.method = method(
			model = model,
        	input_dim=[(3,self.map_size,self.map_size),(2,7)],
        	output_dim=[2,self.map_size*self.map_size,self.map_size*self.map_size],
        	pi_lr=pi_lr,
        	gamma=gamma,
        	buffer_size=buffer_size,
			clipping_range = clipping_range,
			beta = beta
		)


		# Load the model
		if load_model:
            #Load the existing model
			saves = [int(x[len(self.model_name):x.find(".")] )for x in os.listdir(self.path+'/'+self.method_name+'/'+self.model_name+'/actor/') if self.model_name in x[:x.find(".")] and len(x[:x.find(".")]) > len(self.model_name)]
			self.epoch = max(saves)
			path = self.path+'/'+self.method_name+'/'+self.model_name+'/'
			name = self.model_name+str(self.epoch)
			self.method.load(path, name)

		self.logger.drawModel(self.method.model.model, self.path, self.method_name, self.model_name)

	def train(self, obs_new, obs, action, reward):
		# Train the agent
		self.score += reward
		if reward < 0:
			reward = reward * self.coef_neg
		elif reward == 0:
			reward = self.val_null
		else:
			reward = reward * self.coef_pos

		feat = AgentSelectedUnits.get_feature_screen(obs)
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
				path= self.path+'/'+self.method_name+'/'+self.model_name+'/'
				name= self.model_name+str(self.epoch)
				self.method.save(path, name)

	def step(self, obs):
		# step function gets called automatically by pysc2 environment
		# call the parent class to have pysc2 setup rewards/etc for u
		super(AgentSelectedUnits, self).step(obs)
		# if we can move our army (we have something selected)
		# Get the features of the screen
		feat = AgentSelectedUnits.get_feature_screen(obs)
    	# Step with ppo according to this state
		act = self.method.get_action(feat)

		if act[0] == 0:
			if actions.FUNCTIONS.Move_screen.id in obs.observation['available_actions']:
				# Convert the prediction into positions
				position = AgentSelectedUnits.prediction_to_position([act[1]], dim = self.map_size)
				# Get a random location on the map
				return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], position[0]]) , act
			else:
				return actions.FunctionCall(actions.FUNCTIONS.no_op.id,[]), act
		else:
			position1 = AgentSelectedUnits.prediction_to_position([act[1]], dim = self.map_size)
			position2 = AgentSelectedUnits.prediction_to_position([act[2]], dim = self.map_size)
			return actions.FunctionCall(actions.FUNCTIONS.select_rect.id, [[0], position1[0], position2[0]]) , act


	@staticmethod
	def get_feature_screen(obs):
		# Get the feature associated with the observation
		mapp = []
		mapp.append(obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index])
		mapp.append(obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index])
		mapp.append(obs.observation["feature_screen"][features.SCREEN_FEATURES.unit_density.index])

		multi_select = np.zeros((2,7))
		obs_mult = obs.observation["multi_select"]
		for i in range(len(multi_select)):
			if len(obs_mult) > i:
				multi_select[i] = obs_mult[i]
			else:
				break
		return [np.array(mapp),multi_select]

	@staticmethod
	def prediction_to_position(pi, dim):
		# Translate the prediction to y,x position
		pirescale = np.expand_dims(pi, axis=1)
		pirescale = np.append(pirescale, np.zeros_like(pirescale), axis=1)
		positions = np.zeros_like(pirescale)
		positions[:,0] = pirescale[:,0] // dim
		positions[:,1] = pirescale[:,0] % dim
		return positions

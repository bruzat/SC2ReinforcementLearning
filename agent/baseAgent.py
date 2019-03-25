from pysc2.agents import base_agent
from abc import abstractmethod

import agent.log as log

class BaseAgent(base_agent.BaseAgent):
	"""
		An agent for doing a simple movement form one point to another.
	"""

	def __init__(self, model, path='logger/', model_name='model', method_name="method", method=None, load_model=False, val_null=0, coef_neg=1, coef_pos=1, pi_lr=0.001, gamma=0.98, buffer_size=1024, clipping_range=0.2, beta=1e-3):
		super().__init__()
		self.logger = log.Logger()
		self.model_name = model_name
		self.method_name = method_name
		self.nb_steps = 0
		self.max_steps = buffer_size
		self.epoch = 0
		self.path = path
		self.score = 0
		self.score_reset = 0
		self.pi_lr=pi_lr
		self.gamma=gamma
		self.buffer_size = buffer_size
		self.clipping_range = clipping_range
		self.beta = beta
		self.coef_neg = coef_neg
		self.coef_pos = coef_pos
		self.val_null = val_null

	@abstractmethod
	def train(self, obs_new, obs, action, reward):
		pass

	@abstractmethod
	def step(self, obs):
		pass

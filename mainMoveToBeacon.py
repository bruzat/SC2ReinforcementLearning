from tensorflow import keras as k
from agent import agent
from method import policyGradient, trustRegionPolicyOptomisation
from model import simpleDense, multiDense, simpleConv

import argparse
import sys
import os
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features

dict_model = 	{'simpleDense':simpleDense.SimpleDense,
				'multiDense':multiDense.MultiDense,
				'simpleConv':simpleConv.SimpleConv}
dict_method = { 'pg': policyGradient.PolicyGradient,
				 'trpo': trustRegionPolicyOptomisation.TrustRegionPolicyOptomisation}

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, nargs='?', const='model', default='model', help='Name of model')
	parser.add_argument('--method', type=str, nargs='?', const='method', default='methode', help='Name of method')
	parser.add_argument('--load_model', type=bool, help='if load trained model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, nargs='?', const=True, default=True, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	args, unknown_flags = parser.parse_known_args()

	model_name = args.model_name
	method_name = args.method
	visualize = args.visualize
	replay = args.replay
	is_training = args.training
	load_model = args.load_model

	if model_name in dict_model:
		model = dict_model[model_name]
	else:
		model = dict_model['simpleDense']
	print("model is : " + str(model))
	model = model()

	if method_name in dict_method:
		method = dict_method[method_name]
	else:
		method = dict_method['pg']

	print("method is : " + str(method))

	step_mul = 16 if model_name is None else 16
	save_replay_episodes = 10 if replay else 0

	ag = agent.Agent(path='./logger/MoveToBeacon', model_name=model_name, model = model, load_model=load_model,
	 				method_name=method_name, method = method)

	try:
		with sc2_env.SC2Env(map_name="MoveToBeacon", players=[sc2_env.Agent(sc2_env.Race.zerg)], agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=64, minimap=64),
			use_feature_units=True),
			step_mul=step_mul, # Number of step before to ask the next action to from the agent
			visualize=visualize,
			save_replay_episodes=save_replay_episodes,
			replay_dir=os.path.dirname(os.path.abspath(__file__)),
			) as env:

			for i in range(100000):
				ag.setup(env.observation_spec(), env.action_spec())
				timesteps = env.reset()
				ag.reset()
				timesteps = env.step([actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])])
				while True:
					action = ag.step(timesteps[0])
					step_actions = [action]
					old_timesteps = timesteps
					timesteps = env.step(step_actions)
					if(is_training):
						ag.train(timesteps[0], old_timesteps[0],action.arguments[1], timesteps[0].reward)
					if timesteps[0].last():
						break

	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, nargs='?', const='model', default='model', help='Name of model')
	parser.add_argument('--method', type=str, nargs='?', const='methode', default='method', help='Name of methode')
	parser.add_argument('--load_model', type=bool, help='if load trained model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, nargs='?', const=True, default=True, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)
	app.run(main, argv=sys.argv[:1] + unknown_flags)

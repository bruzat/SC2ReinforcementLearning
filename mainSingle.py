from tensorflow import keras as k
from agent import agentSelectedUnits, agentSimple, agentAttMap
from method import policyGradient, trustRegionPolicyOptimization, proximalPolicyOptimization
from model import simpleDense, multiDense, simpleConv, multiConv, spCMS


import argparse
import sys
import os
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features

dict_model = 	{'simpleDense': simpleDense.SimpleDense,
				'multiDense': multiDense.MultiDense,
				'simpleConv': simpleConv.SimpleConv,
				'multiConv': multiConv.MultiConv,
				'spCMS': spCMS.SpCMS}

dict_method = { 'pg': policyGradient.PolicyGradient,
				 'trpo': trustRegionPolicyOptimization.TrustRegionPolicyOptimization,
				 'ppo': proximalPolicyOptimization.ProximalPolicyOptimization}

dict_map = {'MoveToBeacon': 'MoveToBeacon',
			'CollectMineralShards': 'CollectMineralShards',
			'FindAndDefeatZerglings': 'FindAndDefeatZerglings'}

dict_agent = {'simple': agentSimple.AgentSimple,
			'selectedUnits': agentSelectedUnits.AgentSelectedUnits,
			'attMap': agentAttMap.AgentAttMap}

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--map', type=str, default='MoveToBeacon', help='Name of map')
	parser.add_argument('--agent', type=str, default='simple', help='Name of agent')
	parser.add_argument('--model', type=str, default='model', help='Name of model')
	parser.add_argument('--method', type=str, default='method', help='Name of methode')
	parser.add_argument('--load_model', type=bool, help='if load trained model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--no_training', action='store_false', default=True, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	parser.add_argument('--logger_path', type=str, default='./logger', help='path to save log')
	parser.add_argument('--step_mul', type=int, default=8, help='step_mul for pysc2 env')
	args, unknown_flags = parser.parse_known_args()

	model_name = args.model
	method_name = args.method
	visualize = args.visualize
	replay = args.replay
	is_training = args.no_training
	load_model = args.load_model
	logger_path = args.logger_path
	map_name = args.map
	agent_name = args.agent
	step_mul = args.step_mul

	if map_name in dict_map:
		map = dict_map[map_name]
	else:
		map = dict_map['MoveToBeacon']
	print("map is : " + str(map))


	if agent_name in dict_agent:
		agent = dict_agent[agent_name]
	else:
		agent = dict_agent['simple']
	print("agent is : " + str(agent))


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


	save_replay_episodes = 10 if replay else 0

	ag = agent(path=logger_path+'/'+map, model_name=model_name, model = model, load_model=load_model,
	 				method_name=method_name, method = method)

	try:
		with sc2_env.SC2Env(map_name=map, players=[sc2_env.Agent(sc2_env.Race.zerg)], agent_interface_format=features.AgentInterfaceFormat(
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
					action, action_output = ag.step(timesteps[0])
					step_actions = [action]
					old_timesteps = timesteps
					timesteps = env.step(step_actions)
					if(is_training):
						ag.train(timesteps[0], old_timesteps[0],action_output, timesteps[0].reward)
					if timesteps[0].last():
						break



	except KeyboardInterrupt:
		pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--map', type=str, default='MoveToBeacon', help='Name of map')
	parser.add_argument('--agent', type=str, default='simple', help='Name of agent')
	parser.add_argument('--model', type=str, default='model', help='Name of model')
	parser.add_argument('--method', type=str, default='method', help='Name of methode')
	parser.add_argument('--load_model', type=bool, help='if load trained model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--no_training', action='store_false', default=True, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	parser.add_argument('--logger_path', type=str, default='./logger', help='path to save log')
	parser.add_argument('--step_mul', type=int, default=8, help='step_mul for pysc2 env')
	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)
	app.run(main, argv=sys.argv[:1] + unknown_flags)

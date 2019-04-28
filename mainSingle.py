from tensorflow import keras as k
from agent import agentSelectedUnits, agentSimple, agentAttMap, agentSelectAtt, agentRessource
from method import policyGradient, trustRegionPolicyOptimization, proximalPolicyOptimization
from model import simpleDense, multiDense, simpleConv, multiConv


import argparse
import sys
import os
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features

dict_model = 	{'simpleDense': simpleDense.SimpleDense,
				'multiDense': multiDense.MultiDense,
				'simpleConv': simpleConv.SimpleConv,
				'multiConv': multiConv.MultiConv}

dict_method = { 'pg': policyGradient.PolicyGradient,
				 'trpo': trustRegionPolicyOptimization.TrustRegionPolicyOptimization,
				 'ppo': proximalPolicyOptimization.ProximalPolicyOptimization}

dict_map = {'MoveToBeacon': 'MoveToBeacon',
			'CollectMineralShards': 'CollectMineralShards',
			'FindAndDefeatZerglings': 'FindAndDefeatZerglings',
			'DefeatRoaches': 'DefeatRoaches',
			'DefeatZerglingsAndBanelings': 'DefeatZerglingsAndBanelings',
			'CollectMineralsAndGas': 'CollectMineralsAndGas',
			'BuildMarines': 'BuildMarines'}

dict_agent = {'simple': agentSimple.AgentSimple,
			'selectedUnits': agentSelectedUnits.AgentSelectedUnits,
			'attMap': agentAttMap.AgentAttMap,
			'selectAtt': agentSelectAtt.AgentSelectAtt,
			'ressource': agentRessource.AgentRessource}

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

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
	parser.add_argument('--lr', type=float, choices=[Range(0.0, 1.0)], default=0.001, help='learning rate for model Range(0.0, 1.0)')
	parser.add_argument('--gamma', type=float, choices=[Range(0.0, 1.0)], default=0.98, help='gamma for advantage calcul Range(0.0, 1.0)')
	parser.add_argument('--buffer_size', type=int, default=1024, help='buffer size ofr buffer')
	parser.add_argument('--clipping_range', type=float, choices=[Range(0.0, 1.0)], default=0.2, help='clipping_range for model Range(0.0, 1.0) only for ppo')
	parser.add_argument('--beta', type=float, default=1e-3, help='beta for advantage calcul only for ppo')
	parser.add_argument('--coef_neg', type=int,default=1, help='coef mult for negatif reward')
	parser.add_argument('--coef_pos', type=int,default=1, help='coef mult for positif reward')
	parser.add_argument('--val_null', type=int,default=0, help='reward for null value')
	parser.add_argument('--map_size', type=int, default=80, help='size of map')
	parser.add_argument('--minimap_size', type=int, default=64, help='size of minimap')
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
	lr = args.lr
	gamma = args.gamma
	buffer_size = args.buffer_size
	clipping_range = args.clipping_range
	beta = args.beta
	coef_neg = args.coef_neg
	coef_pos = args.coef_pos
	val_null = args.val_null

	minimap_size = args.minimap_size
	map_size = args.map_size

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

	print("lr is : "+str(lr))
	print("gamma is : " +str(gamma))
	print("bugger size is : " +str(buffer_size))
	print("clipping_range is : " +str(clipping_range))
	print("beta size is : " +str(beta))
	print("coef_neg is : " +str(coef_neg))
	print("coef_pos is : " +str(coef_pos))
	print("val_null is : " +str(val_null))

	save_replay_episodes = 10 if replay else 0

	ag = agent(path=logger_path+'/'+map, model_name=model_name, model = model, load_model=load_model,
	 				method_name=method_name, method = method, pi_lr=lr, gamma=gamma, buffer_size=buffer_size,
					clipping_range=clipping_range, beta=beta, coef_neg=coef_neg, coef_pos=coef_pos, val_null=val_null,
					minimap_size=minimap_size, map_size=map_size)

	try:
		with sc2_env.SC2Env(map_name=map, players=[sc2_env.Agent(sc2_env.Race.zerg)], agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=map_size, minimap=minimap_size),
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
	parser.add_argument('--lr', type=float, choices=[Range(0.0, 1.0)], default=0.001, help='learning rate for model Range(0.0, 1.0)')
	parser.add_argument('--gamma', type=float, choices=[Range(0.0, 1.0)], default=0.98, help='gamma for advantage calcul Range(0.0, 1.0)')
	parser.add_argument('--buffer_size', type=int, default=1024, help='buffer size ofr buffer')
	parser.add_argument('--clipping_range', type=float, choices=[Range(0.0, 1.0)], default=0.2, help='clipping_range for model Range(0.0, 1.0) only for ppo')
	parser.add_argument('--beta', type=float, default=1e-3, help='beta for advantage calcul only for ppo')
	parser.add_argument('--coef_neg', type=int,default=1, help='coef mult for negatif reward')
	parser.add_argument('--coef_pos', type=int,default=1, help='coef mult for positif reward')
	parser.add_argument('--val_null', type=int,default=0, help='reward for null value')
	parser.add_argument('--map_size', type=int, default=80, help='size of map')
	parser.add_argument('--minimap_size', type=int, default=64, help='size of minimap')

	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)
	app.run(main, argv=sys.argv[:1] + unknown_flags)

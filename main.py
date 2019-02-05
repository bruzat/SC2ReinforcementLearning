from tensorflow import keras as k
from agent import agent
from rl import policyGradient

import argparse
import sys
import os
from absl import app, flags

from pysc2.env import sc2_env
from pysc2.lib import actions, features

def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	args, unknown_flags = parser.parse_known_args()

	model = args.model
	visualize = args.visualize
	replay = args.replay
	is_training = args.training

	step_mul = 16 if model is None else 16
	save_replay_episodes = 10 if replay else 0

	ag = agent.Agent(model=model, rl = policyGradient.PolicyGradient)

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
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--replay', type=bool, help="Save a replay of the experiment")
	parser.add_argument('--training', type=bool, help="if it is training")
	parser.add_argument('--visualize', type=bool, help="show the agent")
	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)

	app.run(main, argv=sys.argv[:1] + unknown_flags)

import os
import sys

import argparse
from absl import app, flags

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--method', type=str, help='Name of the method')
	parser.add_argument('--map', type=str, help='Name of the method')
	parser.add_argument('--nb', type=int, nargs='?', const=5, default=5, help="number of mean value")
	args, unknown_flags = parser.parse_known_args()

	model = args.model
	method = args.method
	map = args.map
	number_mean = args.nb
	print(number_mean)
	log_train = pd.read_csv('./'+map+'/'+method+'/'+model+'/log.txt')
	log_train.columns = ['Epoch', 'Loss', 'Entropy','MeanReward']


	Loss = []
	Entropy = []
	MeanReward = []
	for i in range(len(log_train)//number_mean):
		l = np.mean(log_train['Loss'][i*number_mean:i*number_mean+number_mean])
		en = np.mean(log_train['Entropy'][i*number_mean:i*number_mean+number_mean])
		mr = np.mean(log_train['MeanReward'][i*number_mean:i*number_mean+number_mean])
		for i in range(number_mean):
			Loss.append(l)
			Entropy.append(en)
			MeanReward.append(mr)


	plt.subplot(3,1,1)
	plt.plot(Loss)
	plt.xlabel('Epoch')
	plt.legend(['Loss'], loc='upper left')

	plt.subplot(3,1,2)
	plt.plot(Entropy)
	plt.xlabel('Epoch')
	plt.legend(['Entropy'], loc='upper left')

	plt.subplot(3,1,3)
	plt.plot(MeanReward)
	plt.xlabel('Epoch')
	plt.legend(['MeanReward'], loc='upper left')
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='Name of the model')
	parser.add_argument('--method', type=str, help='Name of the method')
	parser.add_argument('--map', type=str, help='Name of the method')
	parser.add_argument('--nb', type=bool, nargs='?', const=5, default=5, help="number of mean value")
	args, unknown_flags = parser.parse_known_args()
	flags.FLAGS(sys.argv[:1] + unknown_flags)
	app.run(main, argv=sys.argv[:1] + unknown_flags)

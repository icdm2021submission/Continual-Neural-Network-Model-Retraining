import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns


rewards = []
with open('trainretrain_rewards.log.log') as f:
	lines = f.readlines()
	for line in lines:
		if line.startswith('--------- mini-batch reward:'):
			reward = line.split(':')[-1]
			rewards.append(float(reward.strip()))


print(len(rewards))
print(rewards[0:5])
def plot(values):
	plt.clf()
	colors = ['green', 'green', 'lightblue', 'blue']
	linestyles = ['solid', 'dashed', 'solid', 'solid']
	x = range(0, len(values))
	down = 1
	up = 0
	plt.plot(x, values, color=colors[0], linestyle=linestyles[0], linewidth=2)
	step_down = 0.005
	step_up =0.005	
	plt.xlabel('Mini-batch')
	plt.ylabel('Reward')
	# plt.legend(labels, loc=4)
	# plt.xticks(x, [str(v) for v in x])
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('reward_minibatch_SEA.pdf')

plot(rewards)
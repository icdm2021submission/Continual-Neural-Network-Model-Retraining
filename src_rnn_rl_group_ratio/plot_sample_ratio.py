import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns


plt.close('all')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 12.8)
import seaborn as sns
sns.set(style="white", context="talk")

SMALL_SIZE = 22
MEDIUM_SIZE = 28
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

keys = ['5%', '10%', '20%', '50%']
title = 'The relationship between the test accuracy and the ratio of the samples selected based on Algorithm 1 over the total number of training samples'

def plot(values, title):
	plt.clf()
	colors = ['lightgreen', 'green', 'lightblue', 'blue']
	linestyles = ['solid', 'solid', 'solid', 'solid']
	labels = keys
	x = range(5)
	down = 1
	up = 0
	for i in range(4):
		# print(values[i])
		down = min(down, min(values[i]))
		up = max(down, max(values[i]))
		plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=4)
	step_down = 0.005
	step_up =0.002
	plt.xlabel('Training Session')
	plt.ylabel('Accuracy')
	plt.legend(labels, loc=4)
	plt.xticks(x, ['1', '2', '3', '4', '5'])
	plt.ylim(52, 68)
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('REUTERS_ratio.pdf')


Arms_5 = [53.82, 56.51, 57.99, 57.64, 58.59]
Arms_10 = [58.49 ,58.41 ,58.49 ,59.85, 60.33]
Arms_20 = [58.64, 58.43, 61.37, 60.07, 60.94]
Arms_50 = [59.98, 60.85, 61.85, 61.89, 62.90]
accuracy_np = np.array([Arms_5, Arms_10, Arms_20, Arms_50])
# print(accuracy_np)

plot(accuracy_np, title)		


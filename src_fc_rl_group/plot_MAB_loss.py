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

keys = ['Epoch-loss training', 'Epoch-loss validation', 'MAB-based training', 'MAB-based validation']


def plot(values, title):
	plt.clf()
	colors = ['lightgreen', 'green', 'lightblue', 'blue']
	linestyles = ['dashed', 'dashed', 'solid', 'solid']
	labels = keys
	x = range(1, 20+1)
	down = 1
	up = 0
	for i in range(4):
		down = min(down, min(values[i]))
		up = max(down, max(values[i]))
		plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=4)
	step_down = 0.005
	step_up =0.002
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(labels, loc=4)
	plt.xticks(x, [str(v) for v in x])
	plt.ylim(down-step_down, up+step_up)
	step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_MAB_loss.pdf')

title = 'The loss changes under the MAB-based and the epoch-loss retraining'

# epoch loss
train_loss_list = []
validation_loss_list = []

with open('epoch_loss', 'r') as f:
	lines = f.readlines()
	for line in lines:
		if 'Train metrics: accuracy:' in line:
			train_loss = line.strip().split('loss: ')[-1]
			train_loss_list.append(float(train_loss))
		if 'Eval metrics: accuracy:' in line and 'loss: ' in line:
			validation_loss = line.strip().split('loss: ')[-1]
			# print(line)
			validation_loss_list.append(float(validation_loss))			


# print(train_loss_list)
# print(validation_loss_list)
print(len(train_loss_list))
print(len(validation_loss_list))

# MAB
MAB_train_loss_list = []
MAB_validation_loss_list = []

with open('MAB_loss', 'r') as f:
	lines = f.readlines()
	for line in lines:
		if 'Train metrics: accuracy:' in line:
			train_loss = line.strip().split('loss: ')[-1]
			MAB_train_loss_list.append(float(train_loss))
			# print(line)
		if 'Eval metrics: accuracy:' in line and 'loss: ' in line:
			validation_loss = line.strip().split('loss: ')[-1]
			# print(line)
			MAB_validation_loss_list.append(float(validation_loss))					

# MAB_train_loss_list.extend(MAB_train_loss_list[-1:]*1)
# MAB_validation_loss_list.extend(MAB_validation_loss_list[-1:]*1)
print(MAB_train_loss_list)
print(MAB_validation_loss_list)

print(len(MAB_train_loss_list))
print(len(MAB_validation_loss_list))

accuracy_np = np.array([train_loss_list, validation_loss_list, MAB_train_loss_list, MAB_validation_loss_list])
# print(accuracy_np)

plot(accuracy_np, title)		


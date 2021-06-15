import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns


keys = ['Epoch-based training', 'Epoch-based validation', 'MAB-based training', 'MAB-based validation']


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
		plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
	step_down = 0.005
	step_up =0.005	
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(labels, loc=4)
	plt.xticks(x, [str(v) for v in x])
	plt.ylim(down-step_down, up+step_up)
	step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_MAB_loss.pdf')

title = 'The loss changes under the MAB-based and the epoch-based retraining'

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

train_loss_list.extend(train_loss_list[-1:]*4)
validation_loss_list.extend(validation_loss_list[-1:]*4)
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

# print(MAB_train_loss_list)
# print(MAB_validation_loss_list)

print(len(MAB_train_loss_list))
print(len(MAB_validation_loss_list))

accuracy_np = np.array([train_loss_list, validation_loss_list, MAB_train_loss_list, MAB_validation_loss_list])
# print(accuracy_np)

plot(accuracy_np, title)		


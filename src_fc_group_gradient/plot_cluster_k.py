import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns


keys = ['Training', 'Validation', 'Test']


def plot(values, title):
	plt.clf()
	colors = ['lightgreen', 'lightblue', 'orange']
	linestyles = ['dashed', 'dashed', 'solid']
	labels = keys
	x = range(1, 16)
	down = 1
	up = 0
	for i in range(3):
		down = min(down, min(values[i]))
		up = max(down, max(values[i]))
		plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
	step_down = 0.005
	step_up =0.005	
	plt.xlabel('Number of clusters')
	plt.ylabel('Accuracy')
	plt.legend(labels, loc=4)
	plt.xticks(x, [str(v) for v in x])
	plt.ylim(down-step_down, up+step_up)
	step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_accuracy_cluster_k.pdf')

title = 'Test accuracy corresponds to the (max) number of clusters'

best_train_accuracy_list = []
best_validation_accuracy_list = []
train_accuracy, validation_accuracy = 0, 0
best_train_accuracy, best_validation_accuracy = 0, 0
test_accuracy_list = []
ks = range(1, 16)

for k in ks:
	with open('cluster_{}'.format(k), 'r') as f:
		lines = f.readlines()
		last_line = lines[-1]
		test_accuracy = last_line.split(' ')[-1]
		test_accuracy_list.append(float(test_accuracy))
		for line in lines:
			if 'Train metrics: accuracy:' in line:
				train_accuracy = line.split(' ')[-4]
			if 'Eval metrics: accuracy:' in line:
				validation_accuracy = line.split(' ')[-4]
			if 'Found new best metric score' in line:
				best_train_accuracy, best_validation_accuracy = train_accuracy, validation_accuracy
		best_train_accuracy_list.append(float(best_train_accuracy))
		best_validation_accuracy_list.append(float(best_validation_accuracy))					

# print(best_train_accuracy_list)
# print(best_validation_accuracy_list)
# print(test_accuracy_list)
accuracy_np = np.array([best_train_accuracy_list, best_validation_accuracy_list, test_accuracy_list])
print(accuracy_np)
plot(accuracy_np, title)		


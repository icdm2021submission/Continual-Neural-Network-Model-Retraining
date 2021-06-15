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


# import numpy as np
# import pandas as pd

# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['pdf.fonttype'] = 42
# import matplotlib.pyplot as plt; plt.rcdefaults()
# import seaborn as sns


# plt.close('all')
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (16, 12.8)
# import seaborn as sns
# sns.set(style="white", context="talk")

# SMALL_SIZE = 22
# MEDIUM_SIZE = 28
# BIGGER_SIZE = 28

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

keys = ['Training', 'Validation', 'Test']


def plot(values, num_clusters):
	plt.clf()
	colors = ['lightgreen', 'lightblue', 'orange']
	linestyles = ['dashed', 'dashed', 'solid']
	labels = keys
	x = range(1, num_clusters)
	down = 1
	up = 0
	for i in range(3):
		down = min(down, min(values[i]))
		up = max(down, max(values[i]))
		plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=4)
	step_down = 0.005
	step_up =0.01	
	plt.xlabel('Number of clusters')
	plt.ylabel('Accuracy')
	plt.legend(labels, loc=4)
	# new_xticks = list(range(0, num_clusters, 2))
	# new_xticks = new_xticks[1:]
	# plt.xticks(new_xticks, [str(v) for v in new_xticks])
	plt.xticks(x, [str(v) for v in x])
	plt.ylim(down-step_down, up+step_up)
	step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_accuracy_cluster_k.pdf')

title = 'Training, Validation, and Test accuracy correspond to the (max) number of clusters'
num_clusters = 11
best_train_accuracy_list = []
best_validation_accuracy_list = []
train_accuracy, validation_accuracy = 0, 0
best_train_accuracy, best_validation_accuracy = 0, 0
test_accuracy_list = []
ks = range(1, num_clusters)

for k in ks:
	with open('train_Arms_{}.log'.format(k), 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'Train metrics: accuracy:' in line:
				train_accuracy = line.split(' ')[-4]
			if 'Eval metrics: accuracy:' in line:
				validation_accuracy = line.split(' ')[-4]
			if 'Found new best metric score' in line:
				best_train_accuracy, best_validation_accuracy = train_accuracy, validation_accuracy
		best_train_accuracy_list.append(float(best_train_accuracy))
		best_validation_accuracy_list.append(float(best_validation_accuracy))
	with open('test_Arms_{}.log'.format(k), 'r') as f:
		lines = f.readlines()
		last_line = lines[-2]
		test_accuracy = last_line.split(' ')[-1]
		test_accuracy_list.append(float(test_accuracy))							

# print(best_train_accuracy_list)
# print(best_validation_accuracy_list)
# print(test_accuracy_list)
accuracy_np = np.array([best_train_accuracy_list, best_validation_accuracy_list, test_accuracy_list])
print(accuracy_np)
plot(accuracy_np, num_clusters=num_clusters)		


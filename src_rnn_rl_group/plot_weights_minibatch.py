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
# plt.rcParams["figure.figsize"] = (20,10)
import seaborn as sns

# SMALL_SIZE = 23
# MEDIUM_SIZE = 23
# BIGGER_SIZE = 23

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# (2, 98)
# (200, 98)
# (200,)
# {0: 31, 1: 25, 2: 23, 3: 23, 4: 30, 5: 19, 6: 32, 7: 17}
# (400, 98)
# (52800, 98)
# (52800,)
# {0: 6376, 1: 7909, 2: 8045, 3: 3699, 4: 4491, 5: 8545, 6: 6905, 7: 6830}



layer1 = {0: 202}
layer2 = {0: 31, 1: 25, 2: 23, 3: 23, 4: 30, 5: 19, 6: 32, 7: 17}
layer3 = {0: 6376, 1: 7909, 2: 8045, 3: 3699, 4: 4491, 5: 8545, 6: 6905, 7: 6830}
total = 0
total_dict = layer1
for key, value in layer1.items():
	total += value
for key, value in layer2.items():
	total += value
	if key not in total_dict:
		total_dict[key] = 0
	total_dict[key] += value
for key, value in layer3.items():
	total += value
	if key not in total_dict:
		total_dict[key] = 0	
	total_dict[key] += value

for key, value in total_dict.items():
	total_dict[key] = float(total_dict[key])/total
print(total_dict)

def plot(values, title):
	plt.clf()
	colors = ['lightblue']
	linestyles = ['solid']
	# labels = keys
	# x = range(1, len(values)+1)
	x = range(1, len(values), 2)
	new_values = [values[i] for i in x]	
	down = 1
	up = 0
	plt.plot(x, new_values, color=colors[0], linestyle=linestyles[0], linewidth=4)
	# for i in range(4):
	# 	down = min(down, min(values[i]))
	# 	up = max(down, max(values[i]))
	# 	plt.plot(x, values[i], color=colors[i], linestyle=linestyles[i], linewidth=2)
	step_down = 0.005
	step_up =0.005	
	plt.xlabel('Epoch')
	plt.ylabel('Percentage (%)')
	# plt.legend(labels, loc=4)
	plt.xticks(x, [str(v) for v in x])
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.tight_layout()
	plt.savefig('IMDB_MAB_weight_percentage.pdf')

title = 'The percentage of weights that are optimized'


# MAB
MAB_cluster_id_list = []
percentage_epoch = 0
count = 0
with open('MAB_loss', 'r') as f:
	lines = f.readlines()
	for line in lines:
		if ' [array([' in line:
			cluster_id = line.strip().split('.], dtype=float32)]')[0].split('[array([')[1]
			cluster_id = int(cluster_id)
			percentage = total_dict[cluster_id]
			percentage_epoch += percentage
			count += 1
		if '- Train metrics: accuracy' in line:
			MAB_cluster_id_list.append(100 * percentage_epoch/count)
			count = 0
			percentage_epoch = 0
			# print(cluster_id)				

print(MAB_cluster_id_list)

print(len(MAB_cluster_id_list))

plot(MAB_cluster_id_list, title)


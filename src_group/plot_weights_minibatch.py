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
plt.rcParams["figure.figsize"] = (20,10)
import seaborn as sns

SMALL_SIZE = 28
MEDIUM_SIZE = 28
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# 9
# (800, 9)
# (800,)
# {0: 112, 1: 79, 2: 63, 3: 36, 4: 102, 5: 121, 6: 16, 7: 68, 8: 117, 9: 86}
# 9
# (51200, 9)
# (51200,)
# {0: 5078, 1: 9614, 2: 5308, 3: 7401, 4: 653, 5: 635, 6: 2592, 7: 8166, 8: 9253, 9: 2500}
# 9
# (10240, 9)
# (10240,)
# {0: 660, 1: 1853, 2: 497, 3: 4376, 4: 60, 5: 61, 6: 2733}

layer1 = {0: 112, 1: 79, 2: 63, 3: 36, 4: 102, 5: 121, 6: 16, 7: 68, 8: 117, 9: 86}
layer2 = {0: 5078, 1: 9614, 2: 5308, 3: 7401, 4: 653, 5: 635, 6: 2592, 7: 8166, 8: 9253, 9: 2500}
layer3 = {0: 660, 1: 1853, 2: 497, 3: 4376, 4: 60, 5: 61, 6: 2733}
total = 0
total_dict = layer1
for key, value in layer1.items():
	total += value
	total_dict[key] += 32
total += 32
for key, value in layer2.items():
	total += value
	total_dict[key] += value+64
total += 64
for key, value in layer3.items():
	total += value
	total_dict[key] += value+10
total += 10

for key, value in total_dict.items():
	total_dict[key] = float(total_dict[key])/total
print(total_dict)

def plot(values, title):
	plt.clf()
	colors = ['lightblue']
	linestyles = ['solid']
	# labels = keys
	# x = range(1, len(values)+1, 2)
	x = range(1, len(values), 2)
	new_values = [values[i] for i in x]
	down = 1
	up = 0
	plt.plot(x, new_values, color=colors[0], linestyle=linestyles[0], linewidth=2)
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
	# plt.ylim(0, 30)
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.tight_layout()
	plt.savefig('MNIST_MAB_weight_percentage.pdf')

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


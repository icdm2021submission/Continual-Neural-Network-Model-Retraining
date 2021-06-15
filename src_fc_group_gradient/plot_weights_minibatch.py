import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns


# 2
# (48, 2)
# (48,)
# {0: 4, 1: 11, 2: 8, 3: 1, 4: 2, 5: 3, 6: 10, 7: 4, 8: 4, 9: 1}
# 2
# (64, 2)
# (64,)
# {0: 17, 1: 4, 2: 5, 3: 5, 4: 15, 5: 2, 6: 3, 7: 8, 8: 4, 9: 1}
# 2
# (8, 2)
# (8,)
# {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

layer1 = {0: 4, 1: 11, 2: 8, 3: 1, 4: 2, 5: 3, 6: 10, 7: 4, 8: 4, 9: 1}
layer2 = {0: 17, 1: 4, 2: 5, 3: 5, 4: 15, 5: 2, 6: 3, 7: 8, 8: 4, 9: 1}
layer3 = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
total = 0
total_dict = layer1
for key, value in layer1.items():
	total += value
for key, value in layer2.items():
	total += value
	total_dict[key] += value
for key, value in layer3.items():
	total += value
	total_dict[key] += value

for key, value in total_dict.items():
	total_dict[key] = float(total_dict[key])/total
print(total_dict)

def plot(values, title):
	plt.clf()
	colors = ['lightblue']
	linestyles = ['solid']
	# labels = keys
	x = range(1, len(values)+1)
	down = 1
	up = 0
	plt.plot(x, values, color=colors[0], linestyle=linestyles[0], linewidth=2)
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
	plt.savefig('SEA_MAB_weight_percentage.pdf')

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

print('Done')
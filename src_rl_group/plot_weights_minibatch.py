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
# (800,)
# {0: 90, 1: 99, 2: 69, 3: 61, 4: 100, 5: 105, 6: 89, 7: 124, 8: 27, 9: 36}
# corr_array_shape: (51200, 44)
# (51200,)
# {0: 5917, 1: 5983, 2: 7495, 3: 1624, 4: 7034, 5: 8219, 6: 4291, 7: 5107, 8: 1564, 9: 3966}
# corr_array_shape: (3211264, 44)



# (3211264,)
# {0: 45524, 1: 569059, 2: 233895, 3: 57561, 4: 1023261, 5: 263802, 6: 235931, 7: 582157, 8: 153013, 9: 47061}
# corr_array_shape: (10240, 44)
# (10240,)
# {0: 1553, 1: 928, 2: 572, 3: 2425, 4: 34, 5: 1120, 6: 83, 7: 1497, 8: 1774, 9: 254}
# (5, 5, 1, 32)
# (5, 5, 32, 64)
# (3136, 1024)
# (1024, 10)


layer1 = {0: 93, 1: 199, 2: 201, 3: 76, 4: 231}
layer2 = {0: 13413, 1: 4291, 2: 4375, 3: 16906, 4: 12215}
layer3 = {0: 178347, 1: 427461, 2: 236131, 3: 917863, 4: 1451462}
layer4 = {0: 3151, 1: 3872, 2: 386, 3: 818, 4: 2013}

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

for key, value in layer4.items():
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
	plt.yticks(np.arange(21, 26, step=1))
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
			if cluster_id > 4:
				cluster_id = cluster_id - 5
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


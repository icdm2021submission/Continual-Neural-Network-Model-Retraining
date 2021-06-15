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

# 97
# (2, 97)
# 97
# (200, 97)
# (200,)
# {0: 24, 1: 41, 2: 10, 3: 14, 4: 42, 5: 12, 6: 12, 7: 22, 8: 11, 9: 12}
# 97
# (400, 97)
# 97
# (52800, 97)
# (52800,)
# {0: 7953, 1: 2958, 2: 1894, 3: 8209, 4: 385, 5: 1575, 6: 1497, 7: 1945, 8: 25858, 9: 526}

layer1 = {0: 202, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
layer2 = {0: 24, 1: 41, 2: 10, 3: 14, 4: 42, 5: 12, 6: 12, 7: 22, 8: 11, 9: 12}
layer3 = {0: 7953, 1: 2958, 2: 1894, 3: 8209, 4: 385, 5: 1575, 6: 1497, 7: 1945, 8: 25858, 9: 526}
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
	# x = range(1, len(values)+1)
	x = range(1, 60, 3)
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


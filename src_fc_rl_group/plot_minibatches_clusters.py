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


def plot(values, num_clusters):
	plt.clf()
	x = range(1, num_clusters)
	plt.bar(x, values, align='center', alpha=0.5, linewidth=4)
	plt.xlabel('Cluster')
	plt.ylabel('Count')
	# new_xticks = list(range(0, 21, 2))
	# new_xticks = new_xticks[1:]
	# plt.xticks(new_xticks, [str(v) for v in new_xticks])
	plt.xticks(x, [str(v) for v in x])
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_minibatches_clusters.pdf')

title = 'The number of mini-batches that each cluster selects in a MAB-based retraining session of the SEA dataset'



# MAB
sea_minibatches_clusters = []
total_dict = {}
count = 0
with open('MAB_loss', 'r') as f:
	lines = f.readlines()
	for line in lines:
		if ' [array([' in line:
			cluster_id = line.strip().split('.], dtype=float32)]')[0].split('[array([')[1]
			cluster_id = int(cluster_id)
			if cluster_id not in total_dict:
				total_dict[cluster_id] = 1
			else:
				total_dict[cluster_id] += 1	
sea_minibatches_clusters = total_dict.values()
print(sea_minibatches_clusters)

plot(sea_minibatches_clusters, num_clusters=4)
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
	x = range(1, 11)
	plt.bar(x, values, align='center', alpha=0.5)
	plt.xlabel('Cluster')
	plt.ylabel('The number of mini-batches')
	plt.xticks(x, [str(v) for v in x])
	# plt.ylim(down-step_down, up+step_up)
	# step = round((up-down)/5, 3)
	# plt.yticks(np.arange(down, up, step=step))
	plt.savefig('SEA_minibatches_clusters.pdf')

title = 'The number of mini-batches that each cluster selects in a MAB-based retraining session of the SEA dataset'

sea_minibatches_clusters = [10735, 401, 410, 465, 434, 458, 419, 405, 431, 467]

plot(sea_minibatches_clusters, title)
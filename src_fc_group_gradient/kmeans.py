import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
import glob
sns.set(style="white", context="talk")

def readAllWeights(file_path):
    values = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values.append(float(line.strip()))
    return values

# gap = 10
max_epoch = 0
weight_files = glob.glob('corr_weights1_1_output_*')
for wf in weight_files:
    epoch = int(wf.split('_')[-1])
    max_epoch = max(epoch, max_epoch)

num_docs = max_epoch#10
start_epoch = int(num_docs * 0.8)

weight_names = ['weights1_1', 'weights1_2', 'weights3_2'] #'weights1_1', 'weights1_2', 'weights3_1', 'weights3_2'

for weight_name in weight_names:
    # [# cut-off epoch, # weights]
    epoch_values = []
    for i in range(start_epoch, num_docs):
        values = readAllWeights('./corr_{}_output_{}'.format(weight_name, i))
        nums = len(values)
        # print(nums)
        if nums == 0:
            print(weight_name, str(i))
            continue
        epoch_values.append(values)

    gradient_epoch_values = []
    # # of epochs
    for i in range(1, len(epoch_values)):
        # # of weights
        gradient_values = []
        for j in range(len(epoch_values[i])):
            gradient_values.append(epoch_values[i][j] - epoch_values[i-1][j])
        gradient_epoch_values.append(gradient_values)

    print(len(gradient_epoch_values))
    corr_array = np.array(gradient_epoch_values)
    corr_array = np.transpose(corr_array)
    corr_array_shape = corr_array.shape
    print(corr_array_shape)
    np.save('corr_array_corr', corr_array)

    # (800, 4), (51200, 4), (3211264, 4), (, 4)

    X = np.load('corr_array_corr.npy').reshape(corr_array_shape)
    X = StandardScaler().fit_transform(X)

    num_clusters = 10
    if weight_name == 'weights3_2':
        num_clusters = 7
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    predicted_labels = kmeans.labels_
    cluster_arrays = np.array(predicted_labels)
    print(cluster_arrays.shape)
    # cluster_arrays = np.reshape(-1, 16)
    np.save('cluster_array_{}'.format(weight_name), cluster_arrays)

    unique, counts = np.unique(cluster_arrays, return_counts=True)
    count_per_layer = dict(zip(unique, counts))
    print(count_per_layer)
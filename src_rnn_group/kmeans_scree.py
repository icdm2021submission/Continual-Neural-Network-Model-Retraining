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
sns.set(style="white", context="talk")

def readAllWeights(file_path):
    values = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values.append(float(line.strip()))
    return values

# gap = 10
num_docs = 20
start_epoch = int(num_docs * 0.8)

weight_names = ['rnn_basic_lstm_cell_kernel'] # 'dense_bias', 'dense_kernel', 'rnn_basic_lstm_cell_bias', 'rnn_basic_lstm_cell_kernel'

for weight_name in weight_names:
    # [# cut-off epoch, # weights]
    epoch_values = []
    for i in range(start_epoch, num_docs):
        values = readAllWeights('corrs_20/corr_{}_output_{}'.format(weight_name, i))
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


    # kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    # predicted_labels = kmeans.labels_
    # cluster_arrays = np.array(predicted_labels)
    # print(cluster_arrays.shape)
    # # cluster_arrays = np.reshape(-1, 16)
    # np.save('cluster_array_{}'.format(weight_name), cluster_arrays)


'''
# # #############################################################################
# # Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=50, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = np.transpose(corr_array)
X = StandardScaler().fit_transform(X)
print(type(X))
print(X.shape)
num_vars = X.shape[1]

# Project the data: this step will take several seconds
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(X)
print(digits_proj.shape)
plt.plot(digits_proj[:, 0], digits_proj[:, 1], 'o',
         markeredgecolor='k', markersize=1)
plt.title('TSNE plot for weight values')
plt.show()


#Make a random array and then make it positive-definite
# num_vars = 6
# num_obs = 9
# A = np.random.randn(num_obs, num_vars)
# print(A.shape)
# A = np.asmatrix(A.T) * np.asmatrix(A)
# print(A.shape)
A = np.asmatrix(X.T) * np.asmatrix(X)
print(A.shape)
U, S, V = np.linalg.svd(A) 
eigvals = S**2 / np.sum(S**2)  # NOTE (@amoeba): These are not PCA eigenvalues. 
                               # This question is about SVD.

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it 
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
'''

################-----------------------------------------################

num_clusters = 20
distortions = []
for i in range(1, num_clusters):
    km = KMeans(
        n_clusters=i, init='k-means++',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, num_clusters), distortions, marker='o')
plt.xticks(np.arange(0, 21)) 
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster SSE')
plt.show()

################-----------------------------------------################


# # # #############################################################################
# # # Generate sample data
# # centers = [[1, 1], [-1, -1], [1, -1]]
# # X, labels_true = make_blobs(n_samples=50, centers=centers, cluster_std=0.4,
# #                             random_state=0)
# X = np.transpose(corr_array)
# X = StandardScaler().fit_transform(X)
# # print(type(X))
# # print(X)
# # # #############################################################################
# # # Compute DBSCAN
# db = DBSCAN(eps=0.5, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# # # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# # # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# # # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# # # print("Adjusted Rand Index: %0.3f"
# # #       % metrics.adjusted_rand_score(labels_true, labels))
# # # print("Adjusted Mutual Information: %0.3f"
# # #       % metrics.adjusted_mutual_info_score(labels_true, labels,
# # #                                            average_method='arithmetic'))
# # print("Silhouette Coefficient: %0.3f"
# #       % metrics.silhouette_score(X, labels))

# # #############################################################################
# # Plot result
# import matplotlib.pyplot as plt

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

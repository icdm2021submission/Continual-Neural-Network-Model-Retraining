import numpy as np

num_classes = 2
depth = 3
shape=(int(depth), 16)

array_weights1_1 = np.load('cluster_array_weights1_1.npy')
array_weights1_1 = np.reshape(array_weights1_1, shape)
print(array_weights1_1.shape)
np.save('cluster_array_weights1_1', array_weights1_1)


shape = (16, 4)
array_weights1_2 = np.load('cluster_array_weights1_2.npy')
array_weights1_2 = np.reshape(array_weights1_2, shape)
print(array_weights1_2.shape)
np.save('cluster_array_weights1_2', array_weights1_2)


shape = (4, int(num_classes))
array_weights3_2 = np.load('cluster_array_weights3_2.npy')
array_weights3_2 = np.reshape(array_weights3_2, shape)
print(array_weights3_2.shape)
np.save('cluster_array_weights3_2', array_weights3_2)
import numpy as np

num_classes = 10
depth = 1#mnist is 1, cifar-10 is 3
shape=(5, 5, -1, 32)

array_weights1_1 = np.load('cluster_array_weights1_1.npy')
array_weights1_1 = np.reshape(array_weights1_1, shape)
print(array_weights1_1.shape)
np.save('cluster_array_weights1_1', array_weights1_1)

shape = (5, 5, 32, 64)
array_weights1_2 = np.load('cluster_array_weights1_2.npy')
array_weights1_2 = np.reshape(array_weights1_2, shape)
print(array_weights1_2.shape)
np.save('cluster_array_weights1_2', array_weights1_2)

shape = (-1, 1024)
array_weights3_1 = np.load('cluster_array_weights3_1.npy')
array_weights3_1 = np.reshape(array_weights3_1, shape)
print(array_weights3_1.shape)
np.save('cluster_array_weights3_1', array_weights3_1)

shape = (1024, int(num_classes))
array_weights3_2 = np.load('cluster_array_weights3_2.npy')
array_weights3_2 = np.reshape(array_weights3_2, shape)
print(array_weights3_2.shape)
np.save('cluster_array_weights3_2', array_weights3_2)
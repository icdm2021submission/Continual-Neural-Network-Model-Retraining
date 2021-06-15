import numpy as np

num_classes = 2
depth = 1#mnist is 1, cifar-10 is 3

shape=(2)
array_weights1_1 = np.load('cluster_array_dense_bias.npy')
array_weights1_1 = np.squeeze(array_weights1_1)
print(array_weights1_1.shape)
np.save('cluster_array_dense_bias', array_weights1_1)

shape = (100, 2)
array_weights1_2 = np.load('cluster_array_dense_kernel.npy')
array_weights1_2 = np.reshape(array_weights1_2, shape)
print(array_weights1_2.shape)
np.save('cluster_array_dense_kernel', array_weights1_2)

# shape = (400)
array_weights3_1 = np.load('cluster_array_rnn_basic_lstm_cell_bias.npy')
array_weights3_1 = np.squeeze(array_weights3_1)
print(array_weights3_1.shape)
np.save('cluster_array_rnn_basic_lstm_cell_bias', array_weights3_1)

shape = (132, 400)
array_weights3_2 = np.load('cluster_array_rnn_basic_lstm_cell_kernel.npy')
array_weights3_2 = np.reshape(array_weights3_2, shape)
print(array_weights3_2.shape)
np.save('cluster_array_rnn_basic_lstm_cell_kernel', array_weights3_2)
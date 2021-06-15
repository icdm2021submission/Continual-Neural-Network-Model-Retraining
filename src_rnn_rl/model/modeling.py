"""Define the model."""
import sys, random, logging
import tensorflow as tf
import numpy as np
from util import loss_fns, search_metrics
from tensorflow.python.ops import array_ops

from matplotlib import pyplot
import os
import math
import functools
import time
import kfac

def lstm_mine3(X, is_training, params=None, var_scope='cnn', trainable=True):
    l2_loss = tf.constant(0.0, dtype=tf.float32)
    neurons = []
    weights = []
    old_neurons = []
    gradients_n = []
    gradients_w = []
    # CONVOLUTION 1 - 1
    with tf.name_scope('conv1_1'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                trainable=trainable)
            biases1_1 = tf.get_variable('biases1_1', shape=[32], \
                trainable=trainable)
            # filter1_1 = tf.stop_gradient(filter1_1)
            # biases1_1 = tf.stop_gradient(biases1_1)
            stride = [1,1,1,1]
            conv1_1 = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
            conv1_1 = tf.nn.relu(out1_1)
            old_neurons.append(conv1_1)
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            filter1_1 = tf.multiply(mask_filter1_1, filter1_1)
            mask_biases1_1 = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.zeros_initializer())
            biases1_1 = tf.multiply(mask_biases1_1, biases1_1)
            # filter1_1 = tf.nn.tanh(filter1_1)
            # filter1_1 = tf.nn.relu(filter1_1)
            stride = [1,1,1,1]
            conv1_1 = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
            conv1_1 = tf.nn.relu(out1_1)
            neurons.append(conv1_1)
        l2_loss += tf.nn.l2_loss(mask_filter1_1-1)
        l2_loss += tf.nn.l2_loss(mask_biases1_1)
        # with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        #     out1_1 = tf.layers.batch_normalization(out1_1, training=is_training, name='bn_conv1_1')
        weights.extend([filter1_1, biases1_1])
    # POOL 1
    with tf.name_scope('pool1'):
        pool1_1 = tf.nn.max_pool(conv1_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 name='pool1_1')
        pool1_1_drop = tf.nn.dropout(pool1_1, params.training_keep_prob)
    # CONVOLUTION 1 - 2
    with tf.name_scope('conv1_2'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_2 = tf.get_variable('weights1_2', shape=[5, 5, 32, 64], \
                trainable=trainable)
            biases1_2 = tf.get_variable('biases1_2', shape=[64], \
                trainable=trainable)
            # filter1_2 = tf.stop_gradient(filter1_2)
            # biases1_2 = tf.stop_gradient(biases1_2)
            conv1_2 = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
            conv1_2 = tf.nn.relu(out1_2)       
            old_neurons.append(conv1_2)
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_2 = tf.get_variable('weights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            filter1_2 = tf.multiply(mask_filter1_2, filter1_2)
            mask_biases1_2 = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.zeros_initializer())
            biases1_2 = tf.multiply(mask_biases1_2, biases1_2)           
            # filter1_2 = tf.nn.tanh(filter1_2)
            # filter1_2 = tf.nn.relu(filter1_2)
            conv1_2 = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
            conv1_2 = tf.nn.relu(out1_2)
            neurons.append(conv1_2)
        l2_loss += tf.nn.l2_loss(mask_filter1_2-1)
        l2_loss += tf.nn.l2_loss(mask_biases1_2)
        weights.extend([filter1_2, biases1_2])
        # with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        #     out1_2 = tf.layers.batch_normalization(out1_2, training=is_training, name='bn_conv1_2')
    # POOL 2
    with tf.name_scope('pool2'):
        pool2_1 = tf.nn.max_pool(conv1_2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 name='pool2_1')
        pool2_1_drop = tf.nn.dropout(pool2_1, params.training_keep_prob)
    #FULLY CONNECTED 1
    with tf.name_scope('fc1') as scope:
        pool2_flat = tf.layers.Flatten()(pool2_1_drop)
        dim = pool2_flat.get_shape()[1].value
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            fc1w = tf.get_variable('weights3_1', shape=[dim, 1024], \
                trainable=trainable)
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                trainable=trainable)
            # fc1w = tf.stop_gradient(fc1w)
            # fc1b = tf.stop_gradient(fc1b)
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            old_neurons.append(fc1)         
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc1w = tf.get_variable('weights3_1', shape=[dim, 1024], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc1w = tf.multiply(mask_fc1w, fc1w)
            mask_fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.zeros_initializer())
            fc1b = tf.multiply(mask_fc1b, fc1b)
            # fc1w = tf.nn.tanh(fc1w)
            # fc1w = tf.nn.relu(fc1w)
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            neurons.append(fc1)
        # with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        #     out = tf.layers.batch_normalization(out, training=is_training, name='bn_fc1w')
        l2_loss += tf.nn.l2_loss(mask_fc1w-1)
        l2_loss += tf.nn.l2_loss(mask_fc1b)
        
        fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
        weights.extend([fc1w, fc1b])
    #FULLY CONNECTED 2
    with tf.name_scope('fc2') as scope:
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                trainable=trainable)
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                trainable=trainable)
            # fc2w = tf.stop_gradient(fc2w)
            # fc2b = tf.stop_gradient(fc2b)
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            old_neurons.append(Ylogits)             
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc2w = tf.multiply(mask_fc2w, fc2w)
            mask_fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.zeros_initializer())
            fc2b = tf.multiply(mask_fc2b, fc2b)
            # fc2w = tf.nn.relu(fc2w)
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            neurons.append(Ylogits)            
        l2_loss += tf.nn.l2_loss(mask_fc2w-1)
        l2_loss += tf.nn.l2_loss(mask_fc2b)
        weights.extend([fc2w, fc2b])
        l2_Ylogits = tf.nn.l2_loss(Ylogits)
    gradients_w = []
    for w in weights:
        gradients_w.append(tf.math.abs(tf.gradients(l2_Ylogits, w)))
    for n in neurons:
        gradients_n.append(tf.math.abs(tf.gradients(l2_Ylogits, n)))        
    return Ylogits, (neurons, old_neurons), (gradients_n, gradients_w)


def lstm(sentence, params, var_scope='cnn'):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
        # embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
        #         shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        # initial_state = lstm_cell.zero_state(params.batch_size, dtype=tf.float32)
        # outputs: [batch_size, max_time, cell.output_size]
        # final_states: [batch_size, cell.state_size]
        # outputs shape[64 500 100]
        # last outputs shape[64 100]        
        outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # Compute logits from the output of the LSTM
        # only the final final_states
        # outputs = tf.Print(outputs, [tf.shape(outputs)], 'outputs shape')
        # last_output = tf.Print(outputs[:, -1], [tf.shape(outputs[:, -1])], 'last outputs shape')
        # final_states = tf.Print(state, [tf.shape(state)], 'state shape')
        last_output = outputs[:, -1]
        logits = tf.layers.dense(last_output, params.num_classes)#state[1]
    return logits, final_states

def collect_lstm(inputs, params, var_scope='cnn'):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    features = inputs['features']
    labels = inputs['labels']  
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
        sentence = tf.nn.embedding_lookup(embeddings, features)
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        # initial_state = lstm_cell.zero_state(params.batch_size, dtype=tf.float32)
        # outputs: [batch_size, max_time, cell.output_size]
        # final_states: [batch_size, cell.state_size]
        # outputs shape[64 500 100]
        # last outputs shape[64 100]        
        outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # Compute logits from the output of the LSTM
        # only the final output
        last_output = outputs[:, -1]
        Ylogits = tf.layers.dense(last_output, params.num_classes)
    with tf.variable_scope("mask", reuse=tf.AUTO_REUSE):
        # do not create mask for the embeddings!

        # # Get word embeddings for each token in the sentence
        # embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
        #         shape=[params.vocab_size, params.embedding_size],
        #         initializer=tf.truncated_normal_initializer(stddev=1e-1))
        # sentence = tf.nn.embedding_lookup(embeddings, features)
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        # initial_state = lstm_cell.zero_state(params.batch_size, dtype=tf.float32)
        # outputs: [batch_size, max_time, cell.output_size]
        # final_states: [batch_size, cell.state_size]
        # outputs shape[64 500 100]
        # last outputs shape[64 100]      
        mask_outputs, mask_final_states = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # Compute logits from the output of the LSTM
        # only the final output
        last_mask_output = mask_outputs[:, -1]
        logits = tf.layers.dense(last_mask_output, params.num_classes)   
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=Ylogits)
    loss = tf.reduce_mean(cross_entropy)
    loss = tf.Print(loss, [loss], 'loss in evaluate on train~~\n')
    masks_ops = []
    trained_vars=[v for v in tf.trainable_variables() if 'model/{}'.format(var_scope) \
    in v.name and 'embeddings' not in v.name]
    masks = [v for v in tf.trainable_variables() if 'model/mask' in v.name]
    # trained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/{}'.format(var_scope))
    # masks = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/mask")
    for (var, mask) in zip(trained_vars, masks):
        # var = tf.Print(var, [var.name], message='var name')
        op = tf.assign(mask, mask + cal_gradient(loss, var))
        masks_ops.append(op)
    return logits, masks_ops

def cal_gradient(loss, weight):
    gradient = tf.math.square(tf.gradients(loss, weight))
    return tf.reshape(gradient, tf.shape(weight))

def retrain_lstm(inputs, params=None, var_scope='cnn'):
    sentence = inputs['features']
    labels = inputs['labels']
    trainable = var_scope=='cnn'
    neurons = []
    weights = []
    gradients_n = []
    gradients_w = []
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
        # embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
        #         shape=[params.vocab_size, params.embedding_size])
        sentence = tf.nn.embedding_lookup(embeddings, sentence)
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        outputs, final_states  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # Compute Ylogits from the output of the LSTM
        # only the final output
        last_output = outputs[:, -1]
        Ylogits = tf.layers.dense(last_output, params.num_classes)

    with tf.variable_scope("mask", reuse=tf.AUTO_REUSE):
        # do not create mask for the embeddings!
        # Apply LSTM over the embeddings
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        mask_outputs, mask_state  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        # Compute logits from the output of the LSTM
        # only the final output
        last_mask_output = mask_outputs[:, -1]
        logits = tf.layers.dense(last_mask_output, params.num_classes)

    # neurons = [sentence]
    # neurons.append(last_output)
    neurons = [last_output]
    weights=[v for v in tf.trainable_variables() if 'model/{}'.format(var_scope) \
    in v.name and 'embeddings' not in v.name]
    gradients_w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/mask")
    # gradients_w = [v for v in tf.trainable_variables() if 'model/mask' in v.name]

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=Ylogits)

    if 'fisher' in params.loss_fn:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=Ylogits)
        loss = tf.reduce_mean(cross_entropy)
        # gradients_n.append(tf.math.square(tf.gradients(loss, last_output)))
        for n in neurons:
            gradients_n.append(tf.math.square(tf.gradients(loss, n)))
    else:# for retrain_regu_mas and retrain_regu_mine
        l2_Ylogits = tf.nn.l2_loss(Ylogits)
        # gradients_w = []     
        # for w in weights:
        #     gradients_w.append(tf.math.abs(tf.gradients(l2_Ylogits, w)))
        for n in neurons:
            gradients_n.append(tf.math.abs(tf.gradients(l2_Ylogits, n)))
    return Ylogits, (neurons, weights), (gradients_n, gradients_w)


def retrain_lstm_mini(inputs, params=None, var_scope='cnn'):
    X = inputs['features']
    labels = inputs['labels']
    trainable = var_scope=='cnn'
    neurons = []
    weights = []
    gradients_w = []
    gradients_n = []
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0), )
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
            weights.extend([filter1_1, biases])
            neurons.append(conv1_1)
        # POOL 1
        with tf.name_scope('pool1'):
            pool1_1 = tf.nn.max_pool(conv1_1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool1_1')
            pool1_1_drop = tf.nn.dropout(pool1_1, params.training_keep_prob)
        # CONVOLUTION 1 - 2
        with tf.name_scope('conv1_2'):
            filter1_2 = tf.get_variable('weights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0), )
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
            weights.extend([filter1_2, biases])
            neurons.append(conv1_2)
        # POOL 2
        with tf.name_scope('pool2'):
            pool2_1 = tf.nn.max_pool(conv1_2,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME',
                                     name='pool2_1')
            pool2_1_drop = tf.nn.dropout(pool2_1, params.training_keep_prob)
        #FULLY CONNECTED 1
        with tf.name_scope('fc1') as scope:
            pool2_flat = tf.layers.Flatten()(pool2_1_drop)
            dim = pool2_flat.get_shape()[1].value
            fc1w = tf.get_variable('weights3_1', shape=[dim, 1024], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0), )
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
            weights.extend([fc1w, fc1b])
            neurons.append(fc1)
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0), )
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            weights.extend([fc2w, fc2b])
            neurons.append(Ylogits)
        if 'fisher' in params.loss_fn or 'mine' in params.loss_fn:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=Ylogits)
            loss = tf.reduce_mean(cross_entropy)
            for w in weights:
                gradients_w.append(tf.math.square(tf.gradients(loss, w)))
            for n in neurons:
                gradients_n.append(tf.math.square(tf.gradients(loss, n)))
        else:#retrain_regu_mas
            l2_Ylogits = tf.nn.l2_loss(Ylogits)
            for w in weights:
                gradients_w.append(tf.math.abs(tf.gradients(l2_Ylogits, w)))
            for n in neurons:
                gradients_n.append(tf.math.abs(tf.gradients(l2_Ylogits, n)))    
    return Ylogits, (neurons, weights), (gradients_n, gradients_w)

def build_model(mode, inputs, params):
    """Compute logits of the model
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    Notice:
        !!! when using the build_model mdprank needs a learning_rate around 1e-5 - 1e-7
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test')
    features = inputs['features']

    if params.collect:
        y_conv, masks = collect_lstm(inputs, params, var_scope='cnn')
        inputs['masks'] = masks
        _, _ = lstm(features, params, var_scope='c_cnn')
        return y_conv, 0.0
    
    if params.use_kfac and params.loss_fn != 'retrain_regu_mine3':
        y_conv, _ = lstm(features, params, var_scope='cnn')
        return y_conv, None
    if is_test:
        # if params.loss_fn=='retrain_regu_mine3':
        #     return lstm_mine3(features, False, params, var_scope='cnn')
        return lstm(features, params, var_scope='cnn')
    # not is_test
    if '2' in params.loss_fn:
        if params.use_bn:
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lstm_mini_bn(inputs, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lstm_mini_bn(inputs, params, var_scope='cnn')
        else:
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lstm_mini(inputs, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lstm_mini(inputs, params, var_scope='cnn')
        # weight regulization
        var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
        in zip(old_weights, weights)]
        var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
        var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
        if params.loss_fn=='retrain_regu_mine2':
            if not is_test:
                neuron_mse_list = [(old_neuron - neuron) * (old_neuron - neuron) for (old_neuron, neuron) \
                in zip(old_neurons, neurons)]
                neuron_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_n, neuron_mse_list)]
                neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
                # # weight regulization
                # var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
                # in zip(old_weights, weights)]
                # var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
                # var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
                regulization_loss = 0.001 * neuron_mses + 0.001 * var_mses
                return y_conv, regulization_loss
        if params.loss_fn=='retrain_regu_fisher2':
            regulization_loss = 0.001 * var_mses
            return y_conv, regulization_loss        
    elif 'retrain' in params.loss_fn:
        if params.loss_fn=='retrain_regu_mine3':
            # y_conv, regulization_loss = lstm_mine3(features, is_training, params, var_scope='cnn')
            y_conv, (old_neurons, neurons), (gradients_o_n, gradients_o_w) = lstm_mine3(features, is_training, params, var_scope='cnn')
            neuron_mse_list = [(old_neuron - neuron) * (old_neuron - neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_n, neuron_mse_list)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
            # # weight regulization
            # var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
            # in zip(old_weights, weights)]
            # var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
            # var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.001 * neuron_mses            
            return y_conv, regulization_loss           
        _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lstm(inputs, params, var_scope='c_cnn')
        y_conv, (neurons, weights), _ = retrain_lstm(inputs, params, var_scope='cnn')
        if params.loss_fn=='retrain':
            return y_conv, None
        if params.loss_fn=='retrain_regu':
            neuron_mse_list = [tf.losses.mean_squared_error(old_neuron, neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
            # weight regulization
            var_mse_list = [tf.losses.mean_squared_error(old_var, var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.001 * neuron_mses + 0.001 * var_mses
            return y_conv, regulization_loss
        if params.loss_fn=='retrain_regu_mine':
            neuron_mse_list = [(old_neuron - neuron) * (old_neuron - neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_n, neuron_mse_list)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
            # weight regulization
            var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.001 * neuron_mses + 0.001 * var_mses
            return y_conv, regulization_loss
        if params.loss_fn=='retrain_regu_minen':
            neuron_mse_list = [(old_neuron - neuron) * (old_neuron - neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_n, neuron_mse_list)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
            regulization_loss = 0.001 * neuron_mses
            return y_conv, regulization_loss  
        if params.loss_fn=='retrain_regu_mas':
            var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.001 * var_mses
            return y_conv, regulization_loss           
        if params.loss_fn=='retrain_regu_fishern':
            neuron_mse_list = [(old_neuron - neuron) * (old_neuron - neuron) for (old_neuron, neuron) \
            in zip(old_neurons, neurons)]
            neuron_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_n, neuron_mse_list)]
            neuron_mses = functools.reduce(lambda x,y:x+y, neuron_mse_list) / len(neuron_mse_list)
            regulization_loss = 0.001 * neuron_mses
            return y_conv, regulization_loss                                   
        if params.loss_fn=='retrain_regu_fisher':
            # weight regulization
            var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.001 * var_mses
            return y_conv, regulization_loss
        if params.loss_fn=='retrain_regu_selfless':
            num_samples = tf.shape(features)[0]
            Rssl = tf.constant(0.0, dtype=tf.float32)
            for layer in range(0, len(neurons)-1):
                neurons_l = tf.reshape(tf.multiply(tf.exp(-gradients_o_n[layer]), neurons[layer]), [num_samples, -1])/1000
                num_neuron = tf.shape(neurons_l)[-1]
                coefficient = tf.range(num_neuron)
                coefficient = tf.cast(coefficient, dtype=tf.float32)
                coefficient = coefficient - tf.transpose(coefficient)
                coefficient = tf.exp(-tf.square(coefficient))
                hihj = tf.reduce_sum(tf.multiply(coefficient, tf.matmul(neurons_l, neurons_l, transpose_a=True)))
                hihj -= tf.reduce_sum(tf.matmul(neurons_l, neurons_l, transpose_b=True))#tf.reduce_sum(tf.square(neurons_l))
                Rssl += hihj
            # weight regulization
            var_mse_list = [(old_var - var) * (old_var - var) for (old_var, var) \
            in zip(old_weights, weights)]
            var_mse_list = [tf.reduce_sum(g*n) for (g, n) in zip(gradients_o_w, var_mse_list)]
            var_mses = functools.reduce(lambda x,y:x+y, var_mse_list) / len(var_mse_list)
            regulization_loss = 0.0005 * Rssl + 0.001 * var_mses           
            return y_conv, regulization_loss
    # cnn models including kfac
    y_conv = None
    y_conv, _ = lstm(features, params, var_scope='cnn')
    if is_training:
        _, _ = lstm(features, params, var_scope='c_cnn')
    return y_conv, None

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    is_test = (mode == 'test')
    labels = inputs['labels']
    loss = None
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # Compute the output distribution of the model and the predictions
        predictions, calcualted_loss = build_model(mode, inputs, params)
        if not is_test:
            with tf.name_scope('loss'):
                # calcualted_loss = tf.Print(calcualted_loss, [calcualted_loss], message='calcualted_loss is \n')
                loss = get_loss(predictions, labels, params, calcualted_loss)
                if params.use_regularization:
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss += tf.reduce_sum(reg_losses)
        if is_training:
            if params.use_kfac:
                with tf.name_scope('kfac_optimizer'):
                    # Register loss
                    layer_collection = kfac.LayerCollection()
                    layer_collection.register_softmax_cross_entropy_loss(predictions, reuse=False)
                    # Register layers
                    layer_collection.auto_register_layers()
                    # Construct training ops
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = kfac.PeriodicInvCovUpdateKfacOpt(learning_rate=params.learning_rate, damping=0.001, \
                        batch_size=params.batch_size, layer_collection=layer_collection)
                    train_op = optimizer.minimize(loss, global_step=global_step)
            elif params.use_bn:
                with tf.name_scope('adam_optimizer'):
                    with tf.variable_scope(params.loss_fn, reuse=tf.AUTO_REUSE):
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            global_step = tf.train.get_or_create_global_step()
                            optimizer = tf.train.AdamOptimizer(params.learning_rate)
                            gradients, variables = zip(*optimizer.compute_gradients(loss))
                            gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_value)
                            train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            else:
                with tf.name_scope('adam_optimizer'):                   
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = tf.train.AdamOptimizer(params.learning_rate)
                    gradients, variables = zip(*optimizer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_value)
                    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        with tf.name_scope('accuracy'):
            argmax_predictions = tf.argmax(predictions, 1)
            argmax_labels = tf.argmax(labels, 1)
            correct_prediction = tf.equal(argmax_predictions, argmax_labels)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.mean(accuracy),
        }
        tf.summary.scalar('accuracy', accuracy)
        if not is_test:
            # Summaries for training and validation
            metrics['loss'] = tf.metrics.mean(loss)
            # metrics['calculated_loss'] = tf.reduce_mean(inputs['calculated_loss'])
            tf.summary.scalar('loss', loss)
         
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), \
        tf.local_variables_initializer(), \
        tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec["predictions"] = predictions
    model_spec["accuracy"] = accuracy
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()
    if is_training:
        model_spec['train_op'] = train_op
        model_spec['loss'] = loss
    return model_spec

def get_loss(predicted_scores, labels,
             params, calcualted_loss=None):
    """
    Return loss based on loss_function_str
    Note: this is for models that have real loss functions
    """
    def _cnn():
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=predicted_scores)      
        loss = tf.reduce_mean(cross_entropy)
        return loss
    def _retrain_regu():
        return _cnn() + calcualted_loss

    options = {
            'cnn': _cnn,
            'retrain': _cnn,
            'retrain_regu': _retrain_regu
    }
    loss_function_str = params.loss_fn
    if 'retrain_regu' in params.loss_fn:
        loss_function_str = 'retrain_regu'
    return options[loss_function_str]()

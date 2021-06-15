"""Define the model."""
import sys, random, logging
import tensorflow as tf
import numpy as np
from util import loss_fns, search_metrics
from tensorflow.python.ops import array_ops

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import math
import functools
import time
import kfac
from random import randint

def collect_lenet(inputs, is_training, params=None, var_scope='cnn', trainable=True):
    X = inputs['features']
    labels = inputs['labels']
    # trainable = var_scope == 'cnn'
    # CONVOLUTION 1 - 1
    with tf.name_scope('conv1_1'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            biases1_1 = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0), )        
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_1 = tf.get_variable('mweights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.zeros_initializer(), )
            mask_biases1_1 = tf.get_variable('mbias_1', shape=[32], \
                initializer=tf.zeros_initializer(), )
        stride = [1,1,1,1]
        conv1_1 = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
        out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            # out1_1 = tf.layers.batch_normalization(out1_1, training=is_training, name='bn_conv1_1')
            conv1_1 = tf.nn.relu(out1_1)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            biases1_2 = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0), )            
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_2 = tf.get_variable('mweights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.zeros_initializer(), )
            mask_biases1_2 = tf.get_variable('mbiases1_2', shape=[64], \
                initializer=tf.zeros_initializer(), )          
        conv1_2 = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
        out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            # out1_2 = tf.layers.batch_normalization(out1_2, training=is_training, name='bn_conv1_2')
            conv1_2 = tf.nn.relu(out1_2)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0), )            
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc1w = tf.get_variable('mweights3_1', shape=[dim, 1024], \
                initializer=tf.zeros_initializer(), ) 
            mask_fc1b = tf.get_variable('mbiases3_1', shape=[1024], \
                initializer=tf.zeros_initializer(), )    
        # fc1w = tf.multiply(mask_fc1w, fc1w)
        out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            # out = tf.layers.batch_normalization(out, training=is_training, name='bn_fc1w')
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
    #FULLY CONNECTED 2
    with tf.name_scope('fc2') as scope:
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0), \
                )           
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc2w = tf.get_variable('mweights3_2', shape=[1024, params.num_classes], \
                initializer=tf.zeros_initializer(), \
                )
            mask_fc2b = tf.get_variable('mbiases3_2', shape=[params.num_classes], \
                initializer=tf.zeros_initializer(), \
                )
        Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=Ylogits)
        loss = tf.reduce_mean(cross_entropy)
        loss = tf.Print(loss, [loss], 'loss in evaluate on train~~\n')
        mask_fc2w_op = tf.assign(mask_fc2w, mask_fc2w + cal_gradient(loss, fc2w))
        mask_fc2b_op = tf.assign(mask_fc2b, mask_fc2b + cal_gradient(loss, fc2b))
        mask_fc1w_op = tf.assign(mask_fc1w, mask_fc1w + cal_gradient(loss, fc1w))
        mask_fc1b_op = tf.assign(mask_fc1b, mask_fc1b + cal_gradient(loss, fc1b))
        mask_filter1_2_op = tf.assign(mask_filter1_2, mask_filter1_2 + cal_gradient(loss, filter1_2))
        mask_biases1_2_op = tf.assign(mask_biases1_2, mask_biases1_2 + cal_gradient(loss, biases1_2))
        mask_filter1_1_op = tf.assign(mask_filter1_1, mask_filter1_1 + cal_gradient(loss, filter1_1))
        mask_biases1_1_op = tf.assign(mask_biases1_1, mask_biases1_1 + cal_gradient(loss, biases1_1))
        masks_ops = []
        masks_ops.append(mask_fc2w_op)
        masks_ops.append(mask_fc2b_op)
        masks_ops.append(mask_fc1w_op)
        masks_ops.append(mask_fc1b_op)
        masks_ops.append(mask_filter1_2_op)
        masks_ops.append(mask_biases1_2_op)
        masks_ops.append(mask_filter1_1_op)
        masks_ops.append(mask_biases1_1_op)
    return Ylogits, masks_ops

def cal_gradient(loss, weight):
    gradient = tf.math.square(tf.gradients(loss, weight))
    return tf.reshape(gradient, tf.shape(weight))

def retrain_lenet(inputs, is_training, params=None, var_scope='cnn'):
    X = inputs['features']
    X = tf.cast(X, tf.float32)
    labels = inputs['labels']    
    trainable = var_scope=='cnn'
    neurons = []
    weights = []
    gradients_n = []
    gradients_w = []
    # CONVOLUTION 1 - 1
    with tf.name_scope('conv1_1'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_1 = tf.get_variable('weights1_1', shape=[int(params.depth), 16], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), trainable=trainable)
            biases1_1 = tf.get_variable('biases1_1', shape=[16], \
                initializer=tf.constant_initializer(0.0), trainable=trainable)           
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_1 = tf.get_variable('mweights1_1', shape=[int(params.depth), 16], \
                initializer=tf.zeros_initializer(), trainable=False)
            mask_biases1_1 = tf.get_variable('mbias_1', shape=[16], \
                initializer=tf.zeros_initializer(), trainable=False)
       
        # filter1_1 = tf.multiply(mask_filter1_1, filter1_1)
        # stride = [1,1,1,1]
        conv1_1 = tf.matmul(X, filter1_1)
        out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            # out1_1 = tf.layers.batch_normalization(out1_1, training=is_training, name='bn_conv1_1')
            conv1_1 = tf.nn.relu(out1_1)
            weights.extend([filter1_1, biases1_1])
            neurons.append(conv1_1)
            gradients_w.extend([mask_filter1_1, mask_biases1_1])

        pool1_1_drop = tf.nn.dropout(conv1_1, params.training_keep_prob)
    # CONVOLUTION 1 - 2
    with tf.name_scope('conv1_2'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_2 = tf.get_variable('weights1_2', shape=[16, 4], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                trainable=trainable)
            biases1_2 = tf.get_variable('biases1_2', shape=[4], \
                initializer=tf.constant_initializer(0.0), trainable=trainable)            
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_2 = tf.get_variable('mweights1_2', shape=[16, 4], \
                initializer=tf.zeros_initializer(), trainable=False)
            mask_biases1_2 = tf.get_variable('mbiases1_2', shape=[4], \
                initializer=tf.zeros_initializer(), trainable=False)            
        # filter1_2 = tf.multiply(mask_filter1_2, filter1_2)           
        conv1_2 = tf.matmul(pool1_1_drop, filter1_2)
        out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            # out1_2 = tf.layers.batch_normalization(out1_2, training=is_training, name='bn_conv1_2')
            conv1_2 = tf.nn.relu(out1_2)
            weights.extend([filter1_2, biases1_2])
            neurons.append(conv1_2)
            gradients_w.extend([mask_filter1_2, mask_biases1_2])            

        fc1_drop = tf.nn.dropout(conv1_2, params.training_keep_prob)

    #FULLY CONNECTED 2
    with tf.name_scope('fc2') as scope:
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            fc2w = tf.get_variable('weights3_2', shape=[4, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                trainable=trainable)
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(1.0), \
                trainable=trainable)           
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc2w = tf.get_variable('mweights3_2', shape=[4, params.num_classes], \
                initializer=tf.zeros_initializer(), \
                trainable=False)
            mask_fc2b = tf.get_variable('mbiases3_2', shape=[params.num_classes], \
                initializer=tf.zeros_initializer(), \
                trainable=False)                  
        # fc2w = tf.multiply(mask_fc2w, fc2w)
        Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
        weights.extend([fc2w, fc2b])
        neurons.append(Ylogits)
        gradients_w.extend([mask_fc2w, mask_fc2b])

    if 'fisher' in params.loss_fn:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=Ylogits)
        loss = tf.reduce_mean(cross_entropy)
        for n in neurons:
            gradients_n.append(tf.math.square(tf.gradients(loss, n)))
    else:# for retrain_regu_mas and retrain_regu_mine
        l2_Ylogits = tf.nn.l2_loss(Ylogits)
        gradients_w = []
        for w in weights:
            gradients_w.append(tf.math.abs(tf.gradients(l2_Ylogits, w)))
        for n in neurons:
            gradients_n.append(tf.math.abs(tf.gradients(l2_Ylogits, n)))
    return Ylogits, (neurons, weights), (gradients_n, gradients_w)

def get_random_mask(mask_biases):
    range_mask = tf.ones_like(mask_biases)
    range_mask *= 0.8
    mask_biases = tf.greater(mask_biases, range_mask)
    mask_biases = tf.cast(mask_biases, dtype=tf.float32)
    return mask_biases

def get_mask(mask_biases, action):
    mask_biases = tf.equal(mask_biases, action)
    mask_biases = tf.cast(mask_biases, dtype=tf.float32)
    return mask_biases

def retrain_lenet_bn(inputs, is_training, params=None, var_scope='cnn'):
    X = inputs['features']
    labels = inputs['labels']
    trainable = var_scope=='cnn'
    neurons = []
    weights = []
    gradients_n = []
    gradients_w = []
    # CONVOLUTION 1 - 1
    with tf.name_scope('conv1_1'):
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), )
            biases1_1 = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0), )           
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_1 = tf.get_variable('mweights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.zeros_initializer(), trainable=False)
            mask_biases1_1 = tf.get_variable('mbias_1', shape=[32], \
                initializer=tf.zeros_initializer(), trainable=False)
        # filter1_1 = tf.multiply(mask_filter1_1, filter1_1)
        stride = [1,1,1,1]
        conv1_1 = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
        out1_1 = tf.nn.bias_add(conv1_1, biases1_1)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            out1_1 = tf.layers.batch_normalization(out1_1, training=is_training, name='bn_conv1_1')
            conv1_1 = tf.nn.relu(out1_1)
            weights.extend([filter1_1, biases1_1])
            neurons.append(conv1_1)
            gradients_w.extend([mask_filter1_1, mask_biases1_1])
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            biases1_2 = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0), )            
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_filter1_2 = tf.get_variable('mweights1_2', shape=[5, 5, 32, 64], \
                initializer=tf.zeros_initializer(), trainable=False)
            mask_biases1_2 = tf.get_variable('mbiases1_2', shape=[64], \
                initializer=tf.zeros_initializer(), trainable=False)            
        # filter1_2 = tf.multiply(mask_filter1_2, filter1_2)           
        conv1_2 = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
        out1_2 = tf.nn.bias_add(conv1_2, biases1_2)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            out1_2 = tf.layers.batch_normalization(out1_2, training=is_training, name='bn_conv1_2')
            conv1_2 = tf.nn.relu(out1_2)
            weights.extend([filter1_2, biases1_2])
            neurons.append(conv1_2)
            gradients_w.extend([mask_filter1_2, mask_biases1_2])            
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0), )            
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc1w = tf.get_variable('mweights3_1', shape=[dim, 1024], \
                initializer=tf.zeros_initializer(), trainable=False) 
            mask_fc1b = tf.get_variable('mbiases3_1', shape=[1024], \
                initializer=tf.zeros_initializer(), trainable=False)         
        # fc1w = tf.multiply(mask_fc1w, fc1w)
        out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            out = tf.layers.batch_normalization(out, training=is_training, name='bn_fc1w')
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
            weights.extend([fc1w, fc1b])
            neurons.append(fc1)
            gradients_w.extend([mask_fc1w, mask_fc1b])            
    #FULLY CONNECTED 2
    with tf.name_scope('fc2') as scope:
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1), \
                )
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0), \
                )           
        with tf.variable_scope('mask', reuse=tf.AUTO_REUSE):
            mask_fc2w = tf.get_variable('mweights3_2', shape=[1024, params.num_classes], \
                initializer=tf.zeros_initializer(), \
                trainable=False)
            mask_fc2b = tf.get_variable('mbiases3_2', shape=[params.num_classes], \
                initializer=tf.zeros_initializer(), \
                trainable=False)                  
        # fc2w = tf.multiply(mask_fc2w, fc2w)
        Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
        weights.extend([fc2w, fc2b])
        neurons.append(Ylogits)
        gradients_w.extend([mask_fc2w, mask_fc2b])

    if 'fisher' in params.loss_fn:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=Ylogits)
        loss = tf.reduce_mean(cross_entropy)
        for n in neurons:
            gradients_n.append(tf.math.square(tf.gradients(loss, n)))
    else:# for retrain_regu_mas and retrain_regu_mine
        l2_Ylogits = tf.nn.l2_loss(Ylogits)
        gradients_w = []
        for w in weights:
            gradients_w.append(tf.math.abs(tf.gradients(l2_Ylogits, w)))
        for n in neurons:
            gradients_n.append(tf.math.abs(tf.gradients(l2_Ylogits, n)))
    return Ylogits, (neurons, weights), (gradients_n, gradients_w)

def lenet(X, params=None, var_scope='cnn'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0))
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
    return Ylogits, fc1_drop

def lenet_bn(X, is_training, params=None, var_scope='cnn'):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=is_training, name='bn_conv1_1')
            conv1_1 = tf.nn.relu(out)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=is_training, name='bn_conv1_2')
            conv1_2 = tf.nn.relu(out)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            out = tf.layers.batch_normalization(out, training=is_training, name='bn_fc1w')
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0))
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
    return Ylogits, fc1_drop

def retrain_lenet_mini(inputs, params=None, var_scope='cnn'):
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

def lenet_group(X, params=None, var_scope='cnn', npy_vars=None):
    masks = []
    X = tf.cast(X, tf.float32)
    # action = randint(0, 10)
    with tf.variable_scope('rewards', reuse=tf.AUTO_REUSE):
        action1 = tf.get_variable("action", shape=[1], dtype=tf.float32,
               initializer=tf.zeros_initializer(), trainable=False)
        action = tf.squeeze(action1)
    # action = 0
    # logging.info('*****************************************************************{}'.format(tf.AUTO_REUSE))
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        # CONVOLUTION 1 - 1
        with tf.name_scope('conv1_1'):
            filter1_1 = tf.get_variable('weights1_1', shape=[5, 5, int(params.depth), 32], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            mask_filter1_1 = tf.convert_to_tensor(npy_vars[0], dtype=tf.float32)
            # mask_filter1_1 = tf.Print(mask_filter1_1, [mask_filter1_1], '***************************mask_filter1_1')
            mask_filter1_1 = get_mask(mask_filter1_1, action)
            stride = [1,1,1,1]
            conv = tf.nn.conv2d(X, filter1_1, stride, padding='SAME')
            biases = tf.get_variable('biases1_1', shape=[32], \
                initializer=tf.constant_initializer(0.0))
            mask_biases = tf.ones(shape=[32])
            mask_biases = tf.cast(mask_biases, dtype=tf.float32)
            # mask_biases = get_mask(mask_biases)
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)
            masks.append(mask_filter1_1)
            masks.append(mask_biases)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            mask_filter1_2 = tf.convert_to_tensor(npy_vars[1], dtype=tf.float32)
            # mask_filter1_2 = tf.random_uniform(shape=[5, 5, 32, 64])
            mask_filter1_2 = get_mask(mask_filter1_2, action)
            conv = tf.nn.conv2d(pool1_1_drop, filter1_2, [1,1,1,1], padding='SAME')
            biases = tf.get_variable('biases1_2', shape=[64], \
                initializer=tf.constant_initializer(0.0))
            mask_biases = tf.ones(shape=[64])
            # mask_biases = get_random_mask(mask_biases)
            mask_biases = tf.cast(mask_biases, dtype=tf.float32)   
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)
            masks.append(mask_filter1_2)
            masks.append(mask_biases)
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
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc1b = tf.get_variable('biases3_1', shape=[1024], \
                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(out)
            fc1_drop = tf.nn.dropout(fc1, params.training_keep_prob)
            # mask_fc1w = tf.random_uniform(shape=[dim, 1024])
            mask_fc1w = tf.convert_to_tensor(npy_vars[2], dtype=tf.float32)
            mask_fc1w = get_mask(mask_fc1w, action)
            mask_biases = tf.ones(shape=[1024])
            mask_biases = tf.cast(mask_biases, dtype=tf.float32)
            # mask_biases = get_random_mask(mask_biases)
            masks.append(mask_fc1w)
            masks.append(mask_biases)                   
        #FULLY CONNECTED 2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable('weights3_2', shape=[1024, params.num_classes], \
                initializer=tf.truncated_normal_initializer(stddev=1e-1))
            fc2b = tf.get_variable('biases3_2', shape=[params.num_classes], \
                initializer=tf.constant_initializer(0.0))
            Ylogits = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            mask_fc2w = tf.convert_to_tensor(npy_vars[3], dtype=tf.float32)
            # mask_fc2w = tf.random_uniform(shape=[1024, params.num_classes])
            mask_fc2w = get_mask(mask_fc2w, action)
            mask_biases = tf.ones(shape=[params.num_classes])
            mask_biases = tf.cast(mask_biases, dtype=tf.float32)
            # mask_biases = get_random_mask(mask_biases)
            masks.append(mask_fc2w)
            masks.append(mask_biases)
    return Ylogits, masks

def build_model(mode, inputs, params, npy_vars):
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
    if params.group and not is_test:
        y_conv, masks = lenet_group(features, params, var_scope='cnn', npy_vars=npy_vars)
        return y_conv, masks
    if params.collect:
        if params.use_bn and not params.use_kfac:
            y_conv, masks = collect_lenet_bn(inputs, False, params, var_scope='cnn', trainable=True)
            inputs['masks'] = masks
            _, _ = lenet_bn(features, False, params, var_scope='c_cnn')
        else:
            y_conv, masks = collect_lenet(inputs, False, params, var_scope='cnn', trainable=True)
            inputs['masks'] = masks
            _, _ = lenet(features, params, var_scope='c_cnn')
        return y_conv, 0.0
    
    if params.use_kfac and params.loss_fn != 'retrain_regu_mine3':
        y_conv, _ = lenet(features, params, var_scope='cnn')
        return y_conv, None
    if is_test:
        # if params.loss_fn=='retrain_regu_mine3':
        #     return lenet_mine3(features, False, params, var_scope='cnn')
        if params.use_bn and not params.use_kfac:
            return lenet_bn(features, False, params, var_scope='cnn')
        else:
            return lenet(features, params, var_scope='cnn')
    # not is_test
    if '2' in params.loss_fn:
        if params.use_bn:
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lenet_mini_bn(inputs, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lenet_mini_bn(inputs, params, var_scope='cnn')
        else:
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lenet_mini(inputs, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lenet_mini(inputs, params, var_scope='cnn')
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
            # y_conv, regulization_loss = lenet_mine3(features, is_training, params, var_scope='cnn')
            y_conv, (old_neurons, neurons), (gradients_o_n, gradients_o_w) = lenet_mine3(features, is_training, params, var_scope='cnn')
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
        if params.use_bn:
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lenet_bn(inputs, is_training, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lenet_bn(inputs, is_training, params, var_scope='cnn')
        else:            
            _, (old_neurons, old_weights), (gradients_o_n, gradients_o_w) = retrain_lenet(inputs, is_training, params, var_scope='c_cnn')
            y_conv, (neurons, weights), _ = retrain_lenet(inputs, is_training, params, var_scope='cnn')
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
    if params.use_bn:
        y_conv, _ = lenet_bn(features, is_training, params, var_scope='cnn')
        if is_training:
            _, _ = lenet_bn(features, False, params, var_scope='c_cnn')
    else:
        y_conv, _ = lenet(features, params, var_scope='cnn')
        if is_training:
            _, _ = lenet(features, params, var_scope='c_cnn')
    return y_conv, None

def model_fn(mode, inputs, params, npy_vars=None, reuse=False):
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
        masks = None
        predictions, calcualted_loss = build_model(mode, inputs, params, npy_vars)
        if params.group:
            masks = calcualted_loss
            calcualted_loss = 0.0
        if not is_test:
            with tf.name_scope('loss'):
                # calcualted_loss = tf.Print(calcualted_loss, [calcualted_loss], message='calcualted_loss is \n')
                loss = get_loss(predictions, labels, params, calcualted_loss)
                if params.use_regularization:
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    loss += tf.reduce_sum(reg_losses)
        if is_training:
            if params.group:
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = tf.train.AdamOptimizer(params.learning_rate)
                    gradients, variables = zip(*optimizer.compute_gradients(loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_value)
                    # gradients = [g*m for (g, m) in zip(gradients, masks)]
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

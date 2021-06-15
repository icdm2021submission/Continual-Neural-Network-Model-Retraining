#! /usr/env/bin python3

"""Convert Reuters newswire topics classification Dataset to local TFRecords"""
# https://keras.io/datasets/
import argparse
import os
import sys
import logging
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import reuters
from keras.preprocessing import sequence
from model.utils import save_dict_to_json

max_review_length = 500

def _data_path(data_directory:str, name:str) -> str:
    """Construct a full path to a TFRecord file to be stored in the 
    data_directory. Will also ensure the data directory exists
    
    Args:
        data_directory: The directory where the records will be stored
        name:           The name of the TFRecord
    
    Returns:
        The full path to the TFRecord file
    """
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    return os.path.join(data_directory, f'{name}.tfrecords')

def _int64_feature(value:int) -> tf.train.Features.FeatureEntry:
    """Create a Int64List Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value:str) -> tf.train.Features.FeatureEntry:
    """Create a BytesList Feature
    
    Args:
        value: The value to store in the feature
    
    Returns:
        The FeatureEntry
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images, labels, name:str, data_directory:str, num_shards:int=1):
    """Convert the dataset into TFRecords on disk
    
    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """
    print(f'\nProcessing {name} data')

    # logging.warning('*********************{}'.format(images.shape))
    num_examples, depth = images.shape
    print(num_examples)
    def _process_examples(start_idx:int, end_index:int, filename:str):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(start_idx, end_index):
                sys.stdout.write(f"\rProcessing sample {index+1} of {num_examples}")
                sys.stdout.flush()

                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    # 'height': _int64_feature(rows),
                    # 'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[index])),
                    'image_raw': _bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
    
    if num_shards == 1:
        _process_examples(0, num_examples, _data_path(data_directory, name))
    else:
        total_examples = num_examples
        samples_per_shard = total_examples // num_shards

        for shard in range(num_shards):
            start_index = shard * samples_per_shard
            end_index = start_index + samples_per_shard
            _process_examples(start_index, end_index, _data_path(data_directory, f'{name}-{shard}'))

    return num_examples, depth

def convert_to_tf_record(data_directory:str):
    """Convert the TF MNIST Dataset to TFRecord formats
    
    Args:
        data_directory: The directory where the TFRecord files should be stored
    """
    reuters = keras.datasets.reuters
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=5000)
    logging.warning('********** one text entry: {}'.format(train_data[0]))
    train_data = sequence.pad_sequences(train_data, maxlen=max_review_length)
    test_data = sequence.pad_sequences(test_data, maxlen=max_review_length)
    num_validation_examples, depth = convert_to(train_data, train_labels, 'validation', data_directory)
    num_train_examples, depth = convert_to(train_data, train_labels,  'train', data_directory, num_shards=10)
    num_test_examples, depth = convert_to(test_data, test_labels, 'test', data_directory)
    # Save datasets properties in json file
    sizes = {
        # 'height': rows,
        # 'width': cols,
        'depth': depth,
        'vali_size': num_validation_examples,
        'train_size': num_train_examples,
        'test_size': num_test_examples
    }
    save_dict_to_json(sizes, os.path.join(data_directory, 'dataset_params.json'))   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-directory', 
        default='../data/reuters',
        help='Directory where TFRecords will be stored')

    args = parser.parse_args()
    convert_to_tf_record(os.path.expanduser(args.data_directory))
 
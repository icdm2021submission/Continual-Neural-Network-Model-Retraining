#! /usr/env/bin python3

"""
Convert CIFAR Dataset to local TFRecords
python cifar-to-tfrecords.py --data-directory ../data/cifar-10 --dataset-name cifar-10
python cifar-to-tfrecords.py --data-directory ../data/cifar-100 --dataset-name cifar-100
"""
import json
import argparse
import os
import sys
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from cifar import get_data_set, maybe_download_and_extract

from model.utils import save_data_dict_to_json
import imblearn
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

def get_drif_data_set(data_directory, dataset_name):
    data_path = os.path.join(data_directory, dataset_name)
    with open(data_path) as f:
        lines = f.readlines()
    images, labels = [], []
    for line in lines:
        line = line.strip()
        fields = line.split(',')
        features = fields[0:-1]
        features = [float(v) for v in features]
        images.append(features)
        label = int(fields[-1])
        labels.append(label)
    images = np.array(images)
    images = images.astype('float32')
    print(images.shape)
    return {'images': images, 'labels': labels}

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

def convert_to(data_set, name:str, data_directory:str, mean, std, num_shards:int=1):
    """Convert the dataset into TFRecords on disk
    
    Args:
        data_set:       The MNIST data set to convert
        name:           The name of the data set
        data_directory: The directory where records will be stored
        num_shards:     The number of files on disk to separate records into
    """
    print(f'\nProcessing {name} data')

    images = data_set['images']
    images = (images-mean)/(std+1e-7)
    labels = data_set['labels']
    # logging.warning('*********************', images.shape)
    num_examples, depth = images.shape

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

def convert_to_tf_record(data_directory:str, dataset_name:str):
    """Convert the TF MNIST Dataset to TFRecord formats
    
    Args:
        data_directory: The directory where the TFRecord files should be stored
    """
    dataset_parent_path = "../data"
    # maybe_download_and_extract(dataset_parent_path, dataset_name)
    cifar10 = get_drif_data_set(dataset_parent_path, dataset_name)
    # cifar10_validation = get_data_set(dataset_parent_path, dataset_name, 'validation')
    # cifar10_test = get_data_set(dataset_parent_path, dataset_name, 'test')
    #z-score
    mean = np.mean(cifar10['images'], axis=(0,1))
    std = np.std(cifar10['images'], axis=(0,1))
    total_count, _ = cifar10['images'].shape
    print('/n', cifar10['images'].shape)
    # # x_train = (x_train-mean)/(std+1e-7)
    # # x_test = (x_test-mean)/(std+1e-7)    
    # # num_validation_examples, rows, cols, depth = convert_to(cifar10_validation, 'validation', data_directory, mean, std)
    # num_train_examples = []
    # depth = 3
    # for i in range(0, len(break_points)):
    #     start, end = break_points[i]
    #     start *= 64
    #     end *= 64
    #     if end > total_count:
    #         end = total_count
    #     print(start, end)
    #     ins = {'images': cifar10_train['images'][start: end], 'labels': cifar10_train['labels'][start: end]}
    #     num_train_example, depth = convert_to(ins, 'train-{}'.format(i), data_directory, mean, std)
    #     num_train_examples.append(num_train_example)
    # split_pos = int(total_count*0.8)
    # cifar10_train = {'images': cifar10['images'][0:split_pos, :], 'labels': cifar10['labels'][0:split_pos]}
    # cifar10_test = {'images': cifar10['images'][split_pos:, :], 'labels': cifar10['labels'][split_pos:]}

    x_train, x_test, y_train, y_test = train_test_split(cifar10['images'], cifar10['labels'], test_size = 0.2, random_state = 42)
    cifar10_train = {'images': x_train, 'labels': y_train}
    cifar10_test = {'images': x_test, 'labels': y_test}
    num_train_examples, depth = convert_to(cifar10_train, 'train', data_directory, mean, std, num_shards=10)
    num_validation_examples, depth = convert_to(cifar10_train, 'validation', data_directory, mean, std)
    num_test_examples, depth = convert_to(cifar10_test, 'test', data_directory, mean, std)
    x_test_aug, y_test_aug = oversample.fit_resample(x_test, y_test)
    cifar10_test_aug = {'images': x_test_aug, 'labels': y_test_aug}    
    num_test_aug_examples, depth = convert_to(cifar10_test_aug, 'test_aug', data_directory, mean, std)    
    # Save datasets properties in json file
    sizes = {
        # 'height': rows,
        # 'width': cols,
        'depth': depth,
        'vali_size': num_validation_examples,
        'train_size': num_train_examples,
        'test_size': num_test_examples
    }
    save_data_dict_to_json(sizes, os.path.join(data_directory, 'dataset_params.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data-directory', 
        default='../data/sea',
        help='Directory where TFRecords will be stored')
    parser.add_argument(
        '--dataset-name', 
        default='sea.data',
        help='Directory where TFRecords will be stored')
    args = parser.parse_args()
    # default='../data/sea',
    # default='sea.data',
    convert_to_tf_record(os.path.expanduser(args.data_directory), args.dataset_name)
 
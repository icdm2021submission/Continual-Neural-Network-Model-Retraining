"""Evaluate the model"""

import argparse
import logging
import os
import glob
import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger, load_best_metric, load_learner_id
from model.evaluation import evaluate
from model.reader import input_fn
from model.reader import load_dataset_from_tfrecords
from model.modeling import model_fn
from model.utils import cal_train_size

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--residual_model_dir', default='experiments/residual_model',
                    help="Directory containing params.json")
# loss functions
# cnn, boost, retrain_regu
parser.add_argument('--loss_fn', default='cnn', help="model loss function")
# tf data folder for
# mnist, cifar-10
parser.add_argument('--data_dir', default='../data/mnist',
                    help="Directory containing the dataset")
# test.tfrecords
parser.add_argument('--tfrecords_filename', default='.tfrecords',
                    help="Dataset-filename for the tfrecords")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of the best weights")
parser.add_argument('--train_range', default='[0-4]',
                    help="training tf range")
parser.add_argument('--aug', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="test on augmented test dataset")
parser.add_argument('--combine', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="test on old and augmented test datasets")
parser.add_argument('--finetune', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="finetune mode")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="test log postfix")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.dict['loss_fn'] = args.loss_fn
    params.dict['collect'] = False
    params.dict['use_kfac'] = args.use_kfac
    params.dict['finetune'] = args.finetune  
    params.dict['training_keep_prob'] = 1.0
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build.py".format(json_path)
    params.update(json_path)
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'test{}.log'.format(args.log)))
    # # Get paths for tfrecords
    dataset = 'test'
    if args.combine:
        params.dict['test_size'] = params.dict['test_size'] * 2
        print('USING both Tests')
        logging.info("test size: {}".format(params.test_size))
        dataset += '*'
        path_eval_tfrecords = os.path.join(args.data_dir, dataset + args.tfrecords_filename)
        # Create the input data pipeline
        logging.info("Creating the dataset...")
        eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
    else:
        params.dict['test_size'] = cal_train_size(params.test_size, args.train_range)
        path_eval_tfrecords = os.path.join(args.data_dir, 'test-{}'.format(args.train_range) + args.tfrecords_filename)
        # Create the input data pipeline
        logging.info("Creating the dataset...")
        # eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
        eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
    # Create iterator over the test set
    eval_inputs = input_fn('test', eval_dataset, params)
    logging.info("- done.")
    # Define the model
    logging.info("Creating the model...")
    # weak_learner_id = load_learner_id(os.path.join(args.model_dir, args.restore_from, 'learner.json'))[0]
    eval_model_spec = model_fn('test', eval_inputs, params, reuse=False)
    # node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(node_names)
    logging.info("- done.")
    logging.info("Starting evaluation")
    evaluate(eval_model_spec, args.model_dir, params, args.restore_from)

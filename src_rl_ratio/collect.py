# rm *.txt & ./bash.sh
# experiments/base_model/params.json
# cd /Users/xiaofengzhu/Documents/continual_learning/src
# tensorboard --logdir
import argparse
import logging
import os
import time
import glob
import tensorflow as tf
from model.utils import Params
from model.utils import set_logger
from model.utils import cal_train_size, cal_sample_size
from model.reader import load_dataset_from_tfrecords
from model.reader import input_fn
from model.modeling import model_fn
from model.training import evaluate_on_train


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# loss functions
# cnn, boost, retrain_regu
parser.add_argument('--loss_fn', default='cnn', help="model loss function")
# tf data folder for
# mnist
parser.add_argument('--data_dir', default='../data/mnist',
                    help="Directory containing the dataset")
# test.tfrecords
parser.add_argument('--tfrecords_filename', default='.tfrecords',
                    help="Dataset-filename for the tfrecords")
# usage: python main.py --restore_dir experiments/base_model/best_weights
parser.add_argument('--restore_dir', default=None, # experimens/base_model/best_weights
                    help="Optional, directory containing weights to reload")
parser.add_argument('--train_range', default='[0-4]',
                    help="training tf range")
# using pretrained weights and gradient boosting on datasets A and B
# params.num_learners > 1
# parser.add_argument('--retrain', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
#     help="try on augmented test dataset")
# train on datasets A and B
# parser.add_argument('--collect', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
#     help="try on augmented test dataset")
parser.add_argument('--finetune', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="try on augmented test dataset")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="train log postfix")
parser.add_argument('--cal_sample_ratio', default='5',
                    help="training MAB-based sample ratio")


if __name__ == '__main__':
    # Train the model  
    tf.reset_default_graph()
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)
    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # Set the logger
    set_logger(os.path.join(args.model_dir, 'collect{}.log'.format(args.log)))
    params = Params(json_path)
    params.dict['loss_fn'] = args.loss_fn
    params.dict['collect'] = True
    params.dict['finetune'] = args.finetune
    params.dict['use_kfac'] = args.use_kfac
    params.dict['cal_sample_ratio'] = int(args.cal_sample_ratio)
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, \
    please generate tfrecords".format(json_path)
    params.update(json_path)
    global_epoch = 0
    args.restore_dir = 'best_weights'
    path_train_tfrecords = os.path.join(args.data_dir, 'train-{}'.format(args.train_range) + args.tfrecords_filename)
    path_sample_train_tfrecords = os.path.join(args.data_dir, 'sample' + args.tfrecords_filename)    
    training_files = glob.glob(path_train_tfrecords)
    if args.train_range == '[0-4]':
        params.dict['train_size'] = cal_train_size(params.train_size, args.train_range)
    else:
        params.dict['train_size'] = cal_train_size(params.train_size, args.train_range) + cal_sample_size(params.train_size, args.cal_sample_ratio) # sample.tfrecords is around the same size
        training_files.append(path_sample_train_tfrecords)
        # params.dict['train_size'] = cal_train_size(params.train_size, '[0-' + args.train_range + ']')
    # Create the input data pipeline
    logging.info("Creating the datasets...")
    #########################################################
    params.dict['training_keep_prob'] = 1.0
    start_time = time.time()
    train_dataset = load_dataset_from_tfrecords(training_files)
    # Specify other parameters for the dataset and the model
    # Create the two iterators over the two datasets
    train_inputs = input_fn('vali', train_dataset, params)
    evaluate_on_train_model_spec = model_fn('vali', train_inputs, params, reuse=True)
    logging.info("- done.")
    args.restore_dir = 'best_weights'
    global_epoch = evaluate_on_train(evaluate_on_train_model_spec,
        args.model_dir, params, restore_from=args.restore_dir,\
        global_epoch=global_epoch)
    logging.info("global_epoch: {} epoch(s)".format(global_epoch))
    logging.info("total time: %s seconds ---" % (time.time() - start_time))

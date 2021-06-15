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
from model.utils import cal_train_size
from model.training import train_and_evaluate
from model.reader import load_dataset_from_tfrecords
from model.reader import input_fn
from model.modeling import model_fn
from model.evaluation import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
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
# usage: python main.py --restore_dir experiments/base_model/best_weights
parser.add_argument('--restore_dir', default=None, # experimens/base_model/best_weights
                    help="Optional, directory containing weights to reload")
parser.add_argument('--train_range', default='[0-4]',
                    help="training tf range")
# train on datasets A and B
parser.add_argument('--combine', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="try on augmented test dataset")
parser.add_argument('--finetune', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="try on dataset")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="train log postfix")
parser.add_argument('--rl', default='EXP3',
                    help="rl algorithm")

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
    set_logger(os.path.join(args.model_dir, 'train{}.log'.format(args.log)))    
    params = Params(json_path)
    params.dict['loss_fn'] = args.loss_fn
    params.dict['finetune'] = args.finetune
    params.dict['collect'] = False
    params.dict['use_kfac'] = args.use_kfac
    params.dict['data_dir'] = args.data_dir
    if 'reuters' in args.data_dir:
        params.dict['num_classes'] = 46    
    if args.rl:
        params.dict['rl'] = args.rl
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(args.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, please generate tfrecords".format(json_path)
    params.update(json_path)
    # print(params.dict)
    params.dict['sample_size'] = cal_train_size(params.train_size, '1')
    if args.train_range == '[0-4]':
        params.dict['train_size'] = cal_train_size(params.train_size, args.train_range)
    else:
        params.dict['train_size'] = cal_train_size(params.train_size, args.train_range) + cal_sample_size(params.train_size, args.cal_sample_ratio) # sample.tfrecords is around the same size
    
    last_global_epoch, global_epoch = 0, 0
    if params.num_learners <= 1:# not args.retrain or args.combine:
        if args.combine:
            # train from scratch
            path_train_tfrecords = os.path.join(args.data_dir, 'train-{}'.format(args.train_range) + args.tfrecords_filename)
            # path_eval_tfrecords = os.path.join(args.data_dir, 'validation' + args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation-{}'.format(args.train_range) + args.tfrecords_filename)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
        elif args.finetune:
            args.restore_dir = 'best_weights'
            path_train_tfrecords = os.path.join(args.data_dir, 'train-{}'.format(args.train_range) + args.tfrecords_filename)
            path_sample_train_tfrecords = os.path.join(args.data_dir, 'sample' + args.tfrecords_filename)
            
            # print('path_train_tfrecords: {} ~~~~~~'.format(path_train_tfrecords))
            # 
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation-{}'.format(args.train_range) + args.tfrecords_filename)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            training_files = glob.glob(path_train_tfrecords)
            training_files.append(path_sample_train_tfrecords)
            print('glob.glob(path_train_tfrecords): {} ~~~~~~'.format(training_files))
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            # eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
        else:
            # initial ~ [1-5]
            path_train_tfrecords = os.path.join(args.data_dir, 'train-{}'.format(args.train_range) + args.tfrecords_filename)
            # path_eval_tfrecords = os.path.join(args.data_dir, 'validation' + args.tfrecords_filename)
            path_eval_tfrecords = os.path.join(args.data_dir, 'validation-{}'.format(args.train_range) + args.tfrecords_filename)
            print(path_train_tfrecords)
            # Create the input data pipeline
            logging.info("Creating the datasets...")
            train_dataset = load_dataset_from_tfrecords(glob.glob(path_train_tfrecords))
            # eval_dataset = load_dataset_from_tfrecords(path_eval_tfrecords)
            eval_dataset = load_dataset_from_tfrecords(glob.glob(path_eval_tfrecords))
        # Specify other parameters for the dataset and the model
        # Create the two iterators over the two datasets
        logging.info('train_size: {}'.format(params.train_size))
        train_inputs = input_fn('train', train_dataset, params)
        eval_inputs = input_fn('vali', eval_dataset, params)
        logging.info("- done.")
        # Define the models (2 different set of nodes that share weights for train and validation)
        logging.info("Creating the model...")
        train_model_spec = model_fn('train', train_inputs, params)
        eval_model_spec = model_fn('vali', eval_inputs, params, reuse=True)
        logging.info("- done.")
        logging.info("Starting training for at most {} epoch(s) for the initial learner".format(params.num_epochs))
        start_time = time.time()
        global_epoch = train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, \
            restore_from=args.restore_dir)
        logging.info("global_epoch: {} epoch(s) at learner 0".format(global_epoch))
        logging.info("total time: %s seconds ---" % (time.time() - start_time))

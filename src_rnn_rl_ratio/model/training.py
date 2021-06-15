"""Tensorflow utility functions for training"""
# tensorboard --logdir=experiments/base_model/
# tensorboard --logdir=experiments/base_model/train_summaries
# tensorboard --logdir=experiments/base_model/eval_summaries

import logging
import os

from tqdm import trange
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from model.utils import save_dict_to_json, load_best_metric, get_expaned_metrics
from model.evaluation import evaluate_sess, evaluate_on_train_sess, take_train_samples_sess, evaluate_sess_sample
from model.mabp import rl
import tensorflow.contrib.slim as slim

def train_sess(sess, model_spec, num_steps, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyperparameters
    """
    # Get relevant graph operations or nodes needed for training

    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    # Use tqdm for progress bar
    t = trange(int(num_steps))
    epoch_loss = []
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i == params.save_summary_steps - 1:
        # if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        # t.set_postfix(loss='{:05.3f}'.format(loss_val))
        epoch_loss.append(loss_val)
        # logging.info('loss_val: {}'.format(epoch_loss))
    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val)
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    return epoch_loss

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def isSavingWeights(eval_metrics, best_eval_metrics):
    for i in range(len(eval_metrics)):
        if eval_metrics[i] > best_eval_metrics[i]:
            return True
        elif eval_metrics[i] < best_eval_metrics[i]:
            return False
        else:
            continue
    return False

def evaluate_on_train(eval_model_spec,
    model_dir, params, restore_from, global_epoch=0):
    """Train the model and evaluate every epoch.
    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        sess.run(eval_model_spec['variable_init_op'])
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                begin_at_epoch = int(save_path.split('-')[-1])
                global_epoch = begin_at_epoch + 1       
            logging.info("Restoring parameters from {}".format(save_path))
            # last_saver = tf.train.import_meta_graph(save_path+".meta")
            if params.loss_fn == 'retrain_regu_mine3':
                pretrained_include = ['model/cnn']
            elif params.loss_fn == 'cnn' and params.finetune:
                pretrained_include = ['model/cnn']
            else:
                pretrained_include = ['model/c_cnn']
                pretrained_include.append('model/cnn')
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
        model_summary()
        best_saver = tf.train.Saver(max_to_keep=1)
        # Run one epoch
        logging.info("Epoch {}/{}".format(begin_at_epoch + 1, \
            begin_at_epoch + 1))
        # Compute number of batches in one epoch (one full pass over the training set)
        # Evaluate for one epoch on validation set
        num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
        metrics = evaluate_on_train_sess(sess, eval_model_spec, num_steps, params)
        # loss_evaluate_on_train = sess.run(eval_model_spec['metrics']['loss'])
        # logging.info('loss_evaluate_on_train')
        # print(loss_evaluate_on_train)
        if params.loss_fn == 'cnn' or params.loss_fn == 'retrain_regu':
            cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
            # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
            c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
            update_weights = [tf.assign(c, old) for (c, old) in \
            zip(c_cnn_vars, cnn_vars)]
            sess.run(update_weights)
        # # Save latest eval metrics in a json file in the model directory
        eval_on_train_json_path = os.path.join(model_dir, "metrics_eval_on_train.json")
        save_dict_to_json(metrics, eval_on_train_json_path)
        best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
        best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
        logging.info("- Found new best metric score, saving in {}".format(best_save_path))
    return global_epoch

def choose_best_batches(eval_model_spec,
    model_dir, params, restore_from, sorted_index):
    """Train the model and evaluate every epoch.
    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        sess.run(eval_model_spec['variable_init_op'])
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                begin_at_epoch = int(save_path.split('-')[-1])
                global_epoch = begin_at_epoch + 1       
            logging.info("Restoring parameters from {}".format(save_path))
            # last_saver = tf.train.import_meta_graph(save_path+".meta")
            if params.loss_fn == 'cnn' and params.finetune:
                pretrained_include = ['model/cnn']
            else:
                pretrained_include = ['model/c_cnn']
                pretrained_include.append('model/cnn')
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
        model_summary()
        best_saver = tf.train.Saver(max_to_keep=1)
        # Run one epoch
        logging.info("Epoch {}/{}".format(begin_at_epoch + 1, \
            begin_at_epoch + 1))
        # Compute number of batches in one epoch f(one full pass over the training set)
        # Evaluate for one epoch on validation set
        num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
        metrics = take_train_samples_sess(sess, eval_model_spec, num_steps, params, sorted_index)
        # loss_evaluate_on_train = sess.run(eval_model_spec['metrics']['loss'])
        # logging.info('loss_evaluate_on_train')
        # print(loss_evaluate_on_train)
        if params.loss_fn == 'cnn' or params.loss_fn == 'retrain_regu':
            cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
            # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
            c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
            update_weights = [tf.assign(c, old) for (c, old) in \
            zip(c_cnn_vars, cnn_vars)]
            sess.run(update_weights)
        # # Save latest eval metrics in a json file in the model directory
        eval_on_train_json_path = os.path.join(model_dir, "metrics_eval_on_train.json")
        save_dict_to_json(metrics, eval_on_train_json_path)
        best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
        best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
        logging.info("- Found new best metric score, saving in {}".format(best_save_path))
    return global_epoch

def train_and_evaluate(train_model_spec, eval_model_spec,
    model_dir, params, restore_from=None, global_epoch=1):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])
        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'vali_summaries'), sess.graph)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        best_eval_metrics = [0.0, -float('inf')]
        global_epoch = 0
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                begin_at_epoch = int(save_path.split('-')[-1])
                global_epoch = begin_at_epoch       
            logging.info("Restoring parameters from {}".format(save_path))
            pretrained_include = ['model/cnn']
            if not params.use_kfac:
                pretrained_include.append('model/c_cnn')
            if params.loss_fn != 'cnn':
                pretrained_include.append('model/mask')
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
            # last_best_eval_metric = load_best_metric(best_json_path)
            # best_eval_metrics = [last_best_eval_metric['accuracy'], -last_best_eval_metric['loss']]
            logging.info('best_eval_metrics: {}'.format(best_eval_metrics))            
        model_summary()
        # for each learner
        early_stopping_count = 0
        
        # Compute number of batches in one epoch (one full pass over the training set)
        num_train_steps = (params.train_size + params.batch_size - 1) // params.batch_size
        num_train_steps = int(num_train_steps)
        # Evaluate for one epoch on validation set
        num_vali_steps = (params.vali_size + params.batch_size - 1) // params.batch_size
        num_vali_steps = int(num_vali_steps)
        sum_loss = [0] * num_train_steps
        numbers_of_selections = [0] * num_train_steps
        # UCB specific
        sums_of_reward = [0] * num_train_steps
        arm_weights = [1] * num_train_steps
        # UCB specific
        max_upper_bound = 0
        consk = int(params.consk)
        rounds = round(int(num_train_steps) / consk)
        total_reward = 0
        begin_at_epoch = global_epoch
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if early_stopping_count == int(params.early_stoping_epochs):
                logging.info("Early stopping at epoch {}/{}".format(epoch + 1, \
                    begin_at_epoch + params.num_epochs))
                break
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, \
                begin_at_epoch + params.num_epochs))

            batch_loss = train_sess(sess, train_model_spec, num_train_steps, train_writer, params)
            # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            metrics = evaluate_sess(sess, eval_model_spec, num_vali_steps, eval_writer, params)
            # If best_eval, best_save_path
            accuracy_metric = round(metrics['accuracy'], 6)
            loss_metric = -round(metrics['loss'], 6)
            # save_batch()
            eval_metrics = [accuracy_metric, loss_metric]
            # logging.info('global_epoch: {}, best_eval_metrics: {}, \
            #     eval_metric: {}', global_epoch, best_eval_metrics, eval_metric)
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # and isSavingWeights
                best_eval_metrics = eval_metrics
                # Save weights
                # trainalbe_vars = {v.name: v for v in tf.trainable_variables() if 'model' in v.name}
                # print(trainalbe_vars.keys())
                if params.loss_fn == 'cnn' and not params.use_kfac:
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
        begin_at_epoch = global_epoch
        early_stopping_count = 0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs2):
            # if early_stopping_count == int(params.early_stoping_epochs):
            #     logging.info("Early stopping at epoch {}/{}".format(epoch + 1, \
            #         begin_at_epoch + params.num_epochs))
            #     break
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, \
                begin_at_epoch + params.num_epochs2))

            batch_loss = train_sess(sess, train_model_spec, num_train_steps, train_writer, params)           
            # loss of k consecutive epochs
            # if (epoch - begin_at_epoch + 1) % consk == 0:
            sum_loss = [s+n for (s, n) in zip(batch_loss, sum_loss)]
            sum_loss = [float(v/consk) for v in sum_loss]
            # logging.info('sum_loss :\n {}'.format(sum_loss))
            for i in range(num_train_steps):
                index, reward, numbers_of_selections, sums_of_reward, \
                max_upper_bound = rl(params, sum_loss, numbers_of_selections, \
                    sums_of_reward, max_upper_bound, \
                    (epoch - begin_at_epoch + 1) / consk, arm_weights)          
                # logging.info('numbers_of_selections at i:\n {}'.format(numbers_of_selections))
                total_reward += reward
            # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            metrics = evaluate_sess(sess, eval_model_spec, num_vali_steps, eval_writer, params)
            # If best_eval, best_save_path
            accuracy_metric = round(metrics['accuracy'], 6)
            loss_metric = -round(metrics['loss'], 6)
            # save_batch()
            eval_metrics = [accuracy_metric, loss_metric]
            # logging.info('global_epoch: {}, best_eval_metrics: {}, \
            #     eval_metric: {}', global_epoch, best_eval_metrics, eval_metric)
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # and isSavingWeights
                best_eval_metrics = eval_metrics
                # Save weights
                # trainalbe_vars = {v.name: v for v in tf.trainable_variables() if 'model' in v.name}
                # print(trainalbe_vars.keys())
                if params.loss_fn == 'cnn' and not params.use_kfac:
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
        # logging.info('num_vali_steps: {}'.format(num_vali_steps))
        # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
        # logging.info('numbers_of_selections:\n {}'.format(numbers_of_selections))
        logging.info('numbers_of_selections:\n {}'.format(numbers_of_selections))
        sorted_index = sorted(range(num_train_steps), key=lambda k: numbers_of_selections[k], reverse=True)
        # top_sorted_index = sorted_index[0: int(num_train_steps*params.top_ratio)+1]
        sample_batchs = (params.sample_size + params.batch_size - 1) // params.batch_size
        top_sorted_index = sorted_index[0: int(sample_batchs)+1]
        logging.info('len(top_sorted_index) in training: {}'.format(len(top_sorted_index)))
        take_train_samples_sess(sess, eval_model_spec, num_train_steps, params, top_sorted_index)
    return global_epoch

'''
def train_and_evaluate_sample(train_model_spec, eval_model_spec, eval_train_model_spec,
    model_dir, params, restore_from=None, global_epoch=1):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])
        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'vali_summaries'), sess.graph)
        best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        best_eval_metrics = [0.0, -float('inf')]
        global_epoch = 0
        # Reload weights from directory if specified
        # restor from the previous learner
        if restore_from is not None:
            save_path = os.path.join(model_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
                begin_at_epoch = int(save_path.split('-')[-1])
                global_epoch = begin_at_epoch       
            logging.info("Restoring parameters from {}".format(save_path))
            pretrained_include = ['model/cnn']
            if not params.use_kfac:
                pretrained_include.append('model/c_cnn')
            if params.loss_fn != 'cnn' and params.loss_fn != 'retrain_regu_mine3':
                pretrained_include.append('model/mask')
            pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=pretrained_include)
            pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
            pretrained_saver.restore(sess, save_path)
        model_summary()
        # for each learner
        early_stopping_count = 0
        
        # Compute number of batches in one epoch (one full pass over the training set)
        num_train_steps = (params.train_size + params.batch_size - 1) // params.batch_size
        num_train_steps = int(num_train_steps)
        # Evaluate for one epoch on validation set
        num_vali_steps = (params.vali_size + params.batch_size - 1) // params.batch_size
        num_vali_steps = int(num_vali_steps)
        sum_loss = [0] * num_train_steps
        numbers_of_selections = [0] * num_train_steps
        # UCB specific
        sums_of_reward = [0] * num_train_steps
        arm_weights = [1] * num_train_steps
        # UCB specific
        max_upper_bound = 0
        consk = int(params.consk)
        rounds = round(int(num_train_steps) / consk)
        total_reward = 0

        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            if early_stopping_count == int(params.early_stoping_epochs):
                logging.info("Early stopping at epoch {}/{}".format(epoch + 1, \
                    begin_at_epoch + params.num_epochs))
                break
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, \
                begin_at_epoch + params.num_epochs))

            batch_loss = train_sess(sess, train_model_spec, num_train_steps, train_writer, params)
            # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            metrics = evaluate_sess(sess, eval_model_spec, num_vali_steps, eval_writer, params)
            # If best_eval, best_save_path
            accuracy_metric = round(metrics['accuracy'], 6)
            loss_metric = -round(metrics['loss'], 6)
            # save_batch()
            eval_metrics = [accuracy_metric, loss_metric]
            # logging.info('global_epoch: {}, best_eval_metrics: {}, \
            #     eval_metric: {}', global_epoch, best_eval_metrics, eval_metric)
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # and isSavingWeights
                best_eval_metrics = eval_metrics
                # Save weights
                # trainalbe_vars = {v.name: v for v in tf.trainable_variables() if 'model' in v.name}
                # print(trainalbe_vars.keys())
                if params.loss_fn == 'cnn' and not params.use_kfac:
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/c_cnn')
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
        begin_at_epoch = global_epoch
        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs2):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, \
                begin_at_epoch + params.num_epochs2))

            loss = train_model_spec['loss']
            train_op = train_model_spec['train_op']
            update_metrics = train_model_spec['update_metrics']
            metrics = train_model_spec['metrics']
            summary_op = train_model_spec['summary_op']
            global_step = tf.train.get_global_step()

            # Load the training dataset into the pipeline and initialize the metrics local variables
            # sess.run(train_model_spec['iterator_init_op'])
            sess.run(train_model_spec['metrics_init_op'])
            # Use tqdm for progress bar
            t = range(int(num_train_steps))
            previous_full_loss = 0.0
            for i in t:
                # Evaluate summaries for tensorboard only once in a while
                if i == params.save_summary_steps - 1:
                # if i % params.save_summary_steps == 0:
                    # Perform a mini-batch update
                    _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                      summary_op, global_step])
                    # Write summaries for tensorboard
                    # writer.add_summary(summ, global_step_val)
                else:
                    _, _, loss_val = sess.run([train_op, update_metrics, loss])
                # Log the loss in the tqdm progress bar
                # t.set_postfix(loss='{:05.3f}'.format(loss_val))
                # sum_loss.append(loss_val)
                loss_evaluate_on_train = loss_val
                # logging.info('loss_evaluate_on_one_batch: {}'.format(loss_evaluate_on_train))
                sum_loss[i] = loss_evaluate_on_train
                # metrics = evaluate_sess(sess, eval_model_spec, num_vali_steps, eval_writer, params)
                # loss_evaluate_on_train = evaluate_sess_sample(sess, eval_train_model_spec, num_train_steps, params)
                # logging.info('loss_evaluate_on_train: {}'.format(loss_evaluate_on_train))
                # loss_evaluate_on_train = metrics['loss']
                # sum_loss[i] = abs(loss_evaluate_on_train-previous_full_loss)
                previous_full_loss = loss_evaluate_on_train
                # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
                index, reward, numbers_of_selections, sums_of_reward, \
                max_upper_bound = rl(params, sum_loss, numbers_of_selections, \
                    sums_of_reward, max_upper_bound, \
                    (epoch - begin_at_epoch + 1) / consk, arm_weights)     
                total_reward += reward                
            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
            last_saver.save(sess, last_save_path, global_step=global_epoch)
            # metrics = evaluate_sess(sess, eval_model_spec, num_vali_steps, eval_writer, params)
            # If best_eval, best_save_path
            accuracy_metric = round(metrics['accuracy'], 6)
            loss_metric = -round(metrics['loss'], 6)
            # save_batch()
            eval_metrics = [accuracy_metric, loss_metric]
            # logging.info('global_epoch: {}, best_eval_metrics: {}, \
            #     eval_metric: {}', global_epoch, best_eval_metrics, eval_metric)
            if isSavingWeights(eval_metrics, best_eval_metrics):
                # rest early_stopping_count
                early_stopping_count = 0
                # and isSavingWeights
                best_eval_metrics = eval_metrics
                # # Save weights
                if params.loss_fn == 'cnn' and not params.use_kfac:
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/c_cnn' in v.name]
                    update_weights = [tf.assign(c, old) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                    best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                    best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                    logging.info("- Make a copy of cnn vars, saving in {}".format(best_save_path))             
                elif params.loss_fn == 'retrain_regu_mine3':
                    # c_cnn_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/cnn')
                    c_cnn_vars=[v for v in tf.trainable_variables() if 'model/cnn' in v.name]
                    cnn_vars=[v for v in tf.trainable_variables() if 'model/mask' in v.name]
                    update_weights = [tf.assign(c, tf.multiply(old, c)) for (c, old) in \
                    zip(c_cnn_vars, cnn_vars)]
                    sess.run(update_weights)
                    best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                    best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                    logging.info("- Updated cnn vars, saving in {}".format(best_save_path))
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                # global_epoch = int(params.num_learners) * int(params.num_epochs) + epoch + 1
                best_save_path = best_saver.save(sess, best_save_path, global_step=global_epoch)
                logging.info("- Found new best metric score, saving in {}".format(best_save_path))
                # Save best eval metrics in a json file in the model directory
                save_dict_to_json(metrics, best_json_path)
            else:
                early_stopping_count = early_stopping_count + 1
            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)
            global_epoch += 1
            # # batch_loss = train_sess(sess, train_model_spec, num_train_steps, train_writer, params)
            # # loss of k consecutive epochs
            # # if (epoch - begin_at_epoch + 1) % consk == 0:
            # sum_loss = [s+n for (s, n) in zip(batch_loss, sum_loss)]
            # sum_loss = [float(v/consk) for v in sum_loss]         
        # logging.info('num_vali_steps: {}'.format(num_vali_steps))
        # logging.info('len of sum_loss: {}'.format(len(sum_loss)))
        # logging.info('numbers_of_selections in training: {}'.format(numbers_of_selections))
        sorted_index = sorted(range(num_train_steps), key=lambda k: numbers_of_selections[k], reverse=True)
        # top_sorted_index = sorted_index[0: int(num_train_steps*params.top_ratio)+1]
        sample_batchs = (params.sample_size + params.batch_size - 1) // params.batch_size
        top_sorted_index = sorted_index[0: int(sample_batchs)+1]
        logging.info('len(top_sorted_index) in training: {}'.format(len(top_sorted_index)))
        take_train_samples_sess(sess, eval_model_spec, num_train_steps, params, top_sorted_index)

        global_epoch += 1         
    return global_epoch
'''
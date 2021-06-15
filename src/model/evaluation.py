"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json, save_predictions_to_file, get_expaned_metrics

def evaluate_on_train_sess(sess, model_spec, num_steps, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']

    # global_step = tf.train.get_global_step()
    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    if params.save_predictions:
        # save the predictions and lable_qid to files
        prediction_list = []
        label_list = []
        # compute metrics over the dataset
        for temp_query_id in range(int(num_steps)):
            prediction_per_query, label_per_query, _ = sess.run([model_spec["predictions"], \
                model_spec["labels"], update_metrics])
            prediction_list.extend([v[0] for v in prediction_per_query.tolist()])
            label_per_query_list = label_per_query.tolist()
            label_list.extend(['{} id:{}'.format(int(label_per_query_list[i][0]), \
                temp_query_id) \
                for i in range(0, len(label_per_query_list))])
        save_predictions_to_file(prediction_list, "./prediction_output")
        # tensorflow mess up test input orders
        save_predictions_to_file(label_list, "./label_output")
    else:
        # only update metrics
        for temp_query_id in range(int(num_steps)):
            # # masks_weights=[v for v in tf.trainable_variables() if 'model/mask' in v.name]
            # # # loss = eval_metrics['loss']
            # # # update_masks = [cal_gradient(loss, weight) for weight in masks_weights]
            # # sess.run(update_masks)
            # # sess.run(model_spec['masks'])
            # masks = sess.run(model_spec['masks'])
            # print('Printing masks')
            # print(masks)

            sess.run(model_spec['masks'])
            # mweights3_2=[v for v in tf.trainable_variables() if 'model/mask/mweights3_2' in v.name]
            # mweights3_2 = sess.run(mweights3_2)
            # print('Printing mweights3_2')
            # print(mweights3_2)
            sess.run(update_metrics)
    masks = [v for v in tf.trainable_variables() if 'model/mask' in v.name]
    update_masks = [tf.assign(nm, m/float(num_steps)) for (nm, m) in \
    zip(masks, masks)]
    sess.run(update_masks)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val)
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Collect all train metrics: " + metrics_string)
    return expanded_metrics_val

def cal_gradient(loss, weight):
    gradient = tf.math.square(tf.gradients(loss, weight))
    return tf.reshape(gradient, tf.shape(weight))

def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    accuracy = model_spec['accuracy']
    global_step = tf.train.get_global_step()
    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    # sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])
    sum_eval_accuracy = 0.0
    if params.save_predictions:
        # save the predictions and lable_qid to files
        prediction_list = []
        label_list = []
        # compute metrics over the dataset
        for temp_query_id in range(int(num_steps)):
            # prediction_per_query, label_per_query, height = sess.run([predictions, labels, model_spec["height"]])
            # logging.info("- height per query: \n" + str(height))
            prediction_per_query, label_per_query, _ = sess.run([model_spec["predictions"], \
                model_spec["labels"], update_metrics])
            prediction_list.extend([v[0] for v in prediction_per_query.tolist()])
            # prediction_string = "\n".join(str(v[0]) for v in prediction_per_query.tolist())
            # logging.info("- prediction_per_query: \n" + str(prediction_string))
            label_per_query_list = label_per_query.tolist()
            # label_gains_list = label_gains.tolist()
            # label_per_query_list_string = "\n".join(str(v[0]) for v in label_per_query_list)
            # logging.warning("- label_per_query_list_string: \n" + label_per_query_list_string)
            # label_gains_list_string = "\n".join(str(v[0]) for v in label_gains_list)
            # logging.info("- label_gains_list: \n" + label_gains_list_string)
            label_list.extend(['{} id:{}'.format(int(label_per_query_list[i][0]), \
                temp_query_id) \
                for i in range(0, len(label_per_query_list))])
        save_predictions_to_file(prediction_list, "./prediction_output")
        # tensorflow mess up test input orders
        save_predictions_to_file(label_list, "./label_output")
    else:
        # only update metrics
        for temp_query_id in range(int(num_steps)):
            eval_accuracy, _ = sess.run([accuracy, update_metrics])
            sum_eval_accuracy += eval_accuracy
    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    expanded_metrics_val = get_expaned_metrics(metrics_val)
    metrics_string = " ; ".join("{}: {:05.4f}".format(k, v) for k, v in expanded_metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)
    logging.info("- update_metrics metrics: {}, average accuracy: {:05.4f}".format(metrics_string, sum_eval_accuracy/float(num_steps)))
    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in expanded_metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
    return expanded_metrics_val


def evaluate(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)
    # tf.reset_default_graph()
    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        # Evaluate
        num_steps = (params.test_size + params.batch_size - 1) // params.batch_size
        metrics = evaluate_sess(sess, model_spec, num_steps, params=params)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)

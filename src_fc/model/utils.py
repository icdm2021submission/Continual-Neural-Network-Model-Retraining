"""General utility functions"""

import json
import logging
import re

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        # file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def cal_train_size(params_train_size, args_train_range):
    '''
    args_train_range should follow patterns like [1-5] or 6
    '''
    cuts = 1
    matched = re.match(r"\[(\d+)-(\d+)\]", args_train_range)
    if matched:
        start, end = matched.groups()
        cuts = int(end) - int(start) + 1
    # calculate actual train size
    current_train_size = params_train_size * cuts / 10
    return current_train_size

def save_data_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        json.dump(d, f, indent=4)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_predictions_to_file(l, prediction_output_path):
    """Saves list of floats in txt file

    Args:
        l: (list) of float-castable values (np.float, int, float, etc.)
        prediction_output_path: (string) path to txt file
    """
    with open(prediction_output_path, 'w') as f:
        for v in l:
            f.write(str(v) + '\n')
            f.flush()

def load_best_metric(json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        data = json.load(f)
        return data

def load_learner_id(json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        data = json.load(f)
        return list(data.values())

def get_expaned_metrics(metrics_val):
    expanded_metrics_val = {}
    for tag, val in metrics_val.items():
        if tag == 'accuracy_pc':
            expanded_metrics_val[tag] = str(val)
            # for i in range(len(val)):
            #     expanded_metrics_val['{}_{}'.format(tag, i)] = val[i][0]
        else:
            expanded_metrics_val[tag] = val
    return expanded_metrics_val

"""\
src / labbook.py
--------------------------------------------------------------------------------

Aditya Marathe

Contains tools updating records of my digitial "lab book" for this project. The
lab book keeps a record of the trained Keras models with my own commentary or
observations. 
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'add_log',
]

from typing import Any

import pathlib

import json

from datetime import datetime

import tensorflow as tf
from tensorflow import keras


def _process_dir(
        dir_: str | pathlib.Path
    ) -> pathlib.Path:
    """\
    Converts a directory into a Pathlib Path object, if it is not already one,
    and checks if the directory exists.

    Args
    ----
    dir_: str | pathlib.Path
        The directory/path (usually given by the user).

    Returns
    -------
    pathlib.Path
        The directory as a Path object, if it is confirmed to exist.
    """
    if isinstance(dir_, str):
        dir_ = pathlib.Path(dir_)

    if not dir_.exists():
        raise FileNotFoundError(
            'The directory \'{}\' does not exist.'.format(dir_)
        )

    return dir_


def add_log(
        comments: str,
        config_dict: dict[str, Any],
        train_history: keras.callbacks.History,
        model: keras.Model,
        seralise_objects: dict[str, Any],
        lb_dir: str | pathlib.Path
    ) -> None:
    """\
    Adds a new log to the lab book.

    Args
    ----
    comments: str
        Comments about the model/results - the more specific, the better.
    
    config_dict: dict[str, Any]
        Dictionary containing details about the model and training data 
        configuration. It is required to have the following keys: 'Transforms',
        'XDataCols', and 'YDataCols' otherwise it will raise an Exception.

    train_history: keras.callbacks.History
        Training history of the saved model returned by the `.fit` method.

    model: keras.Model
        The trained model.

    seralise_objects: dict[str, Any]
        A dictionary containing any Python objects or otherwise, that can be 
        seralised using Pickle. This is for saving any additional objects such
        as a StandardScaler object fitted to data.

    lb_dir: str | pathlib.Path
        The path to the labbook.
    """
    # Convert input directory into `Path` object
    lb_dir = _process_dir(lb_dir)

    model_dir = _process_dir(lb_dir / 'models')
    hist_dir = _process_dir(lb_dir / 'history')
    log_dir = _process_dir(lb_dir / 'logs')

    # We do not want to modify the original!
    config_dict = config_dict.copy()

    # Current time for file naming...
    current_time = datetime.now()
    filename_time_string = current_time.strftime('%Y-%m-%d--%H-%M')

    # Does the config. dictionary have the required keys?
    config_require = {
        'Transforms',
        'XDataCols',
        'YDataCols'
    }

    if config_require - set(config_dict.keys()):
        raise ValueError(
            'Log requires the following keys in the `config_dict`: '
            '\'Transforms\', \'XDataCols\', and \'YDataCols\'.'
        )

    # Store a reference to the model's save file and its history, and the log
    config_dict['ModelDir'] = str(
        model_dir.resolve() / f'Model--{filename_time_string}.h5'
    )
    config_dict['HistoryDir'] = str(
        hist_dir.resolve() / f'Hist--{filename_time_string}.json'
    )
    config_dict['LogDir'] = str(
        log_dir.resolve() / f'Log--{filename_time_string}.h5'
    )

    # For when I am looking back at old models after a week...
    config_dict['Time'] = current_time.strftime('%d-%m-%Y %H:%M')
    config_dict['Comments'] = comments
    config_dict['Flagged'] = False

    # Save the model, history and log file...
    model.save(config_dict['ModelDir'])

    with open(config_dict['HistoryDir'], 'w') as file:
        json.dump(train_history.history, file, indent=4)

    with open(config_dict['LogDir'], 'w') as file:
        json.dump(config_dict, file, indent=4)

    # Verbosity for some peace of mind...
    print(
        f'LabBook | {config_dict["Time"]} | Log saved!'
    )


def load_model_from_log(log_file_dir: str | pathlib.Path) -> dict[str, Any]:
    """\
    Loads the saved model from a certain log in the lab book.

    Args
    ----
    log_file_dir: str | pathlib.Path
        The directory to the specific log file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the (keys) 'Model' and 'History'.
    """
    # Convert input directory into `Path` object
    log_file_dir = _process_dir(log_file_dir)

    # Current time for future reference...
    current_time = datetime.now()
    current_time_string = current_time.strftime('%d-%m-%Y %H:%M')

    # Read the log file...
    with open(log_file_dir, 'r') as file:
        log = json.load(file)

    # Load all the stuff
    model = keras.models.load_model(filepath=log['ModelDir'])

    with open(log['HistoryDir'], 'r') as file:
        history = json.load(file)

    # Verbosity
    print(
        f'LabBook | {current_time_string} | Loaded log from {log["Time"]}!'
    )

    # TODO: Also return the seralised stuff...

    return {
        'Model': model,
        'History': history
    }

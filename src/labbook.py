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

import joblib

# import tensorflow as tf
from tensorflow import keras

_CONFIG_REQUIRE = {
    'Transforms',
    'XDataCols',
    'YDataCols'
}


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


def _check_required_keys(config_dict: dict[str, Any]) -> None:
    """\
    Checks if the config. dictionary contains the required keys.
    """
    if _CONFIG_REQUIRE - set(config_dict.keys()):
        raise ValueError(
            'Log requires the following keys in the `config_dict`: '
            ', '.join(list(_CONFIG_REQUIRE)) + '.'
        )


def _serialise(obj: Any, file_dir: str) -> None:
    """\
    Uses `joblib` to serialise objects, namely Sci-kit learn estimators or 
    classifiers.
    """
    joblib.dump(
        value=obj,
        filename=file_dir
    )


def _deserialise(file_dir: str) -> Any:
    return joblib.load(filename=file_dir)


def _serialise_objects(
        objs: dict[str, Any],
        time_str: str,
        pkl_dir: pathlib.Path
    ) -> dict[str, str]:
    """\
    
    """
    serialised_dict = dict()

    for obj_name, obj in objs.items():
        serialised_dict[obj_name] = str(
            pkl_dir.resolve() / f'{obj_name}--{time_str}.pickle'
        )
        _serialise(obj=obj, file_dir=serialised_dict[obj_name])

    return serialised_dict


def _deserialise_objects(file_dir_dict: dict[str, str]) -> dict[str, Any]:
    """\
    
    """
    obj_dict = dict()

    for name, file_dir in file_dir_dict.items():
        obj_dict[name] = _deserialise(file_dir=file_dir)

    return obj_dict


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
    pkl_dir = _process_dir(lb_dir / 'pickled')

    # We do not want to modify the original!
    config_dict = config_dict.copy()

    # Current time for file naming...
    current_time = datetime.now()
    filename_time_string = current_time.strftime('%Y-%m-%d--%H-%M')

    # Does the config. dictionary have the required keys?
    _check_required_keys(config_dict=config_dict)

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

    # Serialise the other stuff as well...
    if seralise_objects is not None:
        config_dict['SerialisedDir'] = _serialise_objects(
            objs=seralise_objects,
            time_str=filename_time_string,
            pkl_dir=pkl_dir
        )

    # Save the model, history and log file...
    model.save(config_dict['ModelDir'])

    with open(config_dict['HistoryDir'], 'w') as file:
        json.dump(train_history.history, file, indent=4)

    with open(config_dict['LogDir'], 'w') as file:
        json.dump(config_dict, file, indent=4)

    # Verbosity for some peace of mind...
    print(f'LabBook  | {config_dict["Time"]} | Log saved!')


def add_log_skl(
        comments: str,
        config_dict: dict[str, Any],
        model: Any,
        lb_dir: str | pathlib.Path,
        seralise_objects: dict[str, Any] | None = None
    ) -> None:
    """\
    Add a new log to the lab book - specifically Sci-kit Learn models.
    """
    # Convert input directory into `Path` object
    lb_dir = _process_dir(lb_dir)

    model_dir = _process_dir(lb_dir / 'models')
    log_dir = _process_dir(lb_dir / 'logs')
    pkl_dir = _process_dir(lb_dir / 'pickled')

    # Does the config. dictionary have the required keys?
    _check_required_keys(config_dict=config_dict)

    # We do not want to modify the original!
    config_dict = config_dict.copy()

    # Current time for file naming...
    current_time = datetime.now()
    filename_time_string = current_time.strftime('%Y-%m-%d--%H-%M')

    # Store a reference to the model's save file and its history, and the log
    config_dict['ModelDir'] = str(
        model_dir.resolve() / f'Model--{filename_time_string}.pickle'
    )
    config_dict['LogDir'] = str(
        log_dir.resolve() / f'Log--{filename_time_string}.h5'
    )

    # For when I am looking back at old models after a week...
    config_dict['Time'] = current_time.strftime('%d-%m-%Y %H:%M')
    config_dict['Comments'] = comments
    config_dict['Flagged'] = False

    # Serialise the other stuff as well...
    if seralise_objects is not None:
        config_dict['SerialisedDir'] = _serialise_objects(
            objs=seralise_objects,
            time_str=filename_time_string,
            pkl_dir=pkl_dir
        )

    # Save the model and log file...
    _serialise(obj=model, file_dir=config_dict['ModelDir'])

    with open(config_dict['LogDir'], 'w') as file:  # type: ignore
        json.dump(config_dict, file, indent=4)

    # Verbosity for some peace of mind...
    print(f'LabBook  | {config_dict["Time"]} | Log saved!')


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
    with open(log_file_dir, 'r') as file:  # type: ignore
        log = json.load(file)

    # Load all the stuff
    model = keras.models.load_model(filepath=log['ModelDir'])

    with open(log['HistoryDir'], 'r') as file:  # type: ignore
        history = json.load(file)

    if log.get('SerialisedDir') is not None:
        serialised = _deserialise_objects(
            file_dir_dict=log['SerialisedDir']
        )

    # Verbosity
    print(f'LabBook  | {current_time_string} | Loaded log from {log["Time"]}!')

    return {
        'Model': model,
        'History': history
    }

"""
src / classification / random_forest.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

import sys
# import argparse

from numpy.random import RandomState

import pandas as pd

import sklearn as skl
from sklearn.ensemble import RandomForestClassifier

import joblib

sys.path.insert(1, './src/')

import ana
import plotting
import utils
import transforms


verbose: bool = True


config = {
    'Cuts': [
        'Detector Quality',
        'Data Quality',
        'Cosmic Rej.',
        'Veto'
    ],
    'Transforms': [
        transforms.tf_290124_positive_energies,
        transforms.tf_290124_valid_pid,
        transforms.tf_120224_first_prong
    ],
    'XCols': [
        'rec.sel.cvnloosepreselptp.cosmicid',
        'rec.sel.cvnloosepreselptp.ncid',
        'rec.sel.cvnloosepreselptp.numuid',
        'rec.sel.remid.pid',
        'rec.sel.scann.scpid'
    ],
    'YCols': [
        'ana.cat.event_type'
    ]
}


def log(message: str = '') -> None:
    """\
    Prints a log message to the consle.
    
    Args
    ----
    message: str
        Log message.
    """
    if verbose:
        print(f'Main     | {message}')


log(f'Sci-kit Learn version {skl.__version__}')


def load_training_data() -> ana.NOvAData:
    """\
    Load NOvA data that is used for training.
    """
    ds = ana.Datasets()
    data = ana.NOvAData.init_from_copymerge_h5(
        h5dirs=[
            ds.COPYMERGED_C10_DIR  # type: ignore
        ]
    )
    data.fill_ana_flags(inplace=True)
    data.fill_ana_track_kinematics(inplace=True)
    data.fill_categorical(inplace=True)

    return data


def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """\
    Data preprocessing pipeline.
        1. Apply the required NOvA cuts.
        2. Apply transformations to the data.
    
    Both steps also clean out nonsense data.
    """
    cuts = ana.Cuts.init_nova_cuts()

    for cut in config['Cuts']:
        data = cuts.apply_cut(cut, data)

    for tf in config['Transforms']:
        data = tf(df=data)
        log(f'Applied a transform which {transforms.get_tf_info(tf=tf)}')

    return data


def get_required_features(data: pd.DataFrame) -> pd.DataFrame:
    """\
    Get the features used for training/testing.
    """
    return data[config['XCols'] + config['YCols']]


def grid_search() -> None:
    return


def main(*a, **kw) -> int:
    """
    Main
    ----

    Pipeline:
        1. Load training data.
        2. Apply preprocessing transforms.
        3. Get the training and testing data.
        4. Build the classifier.
        5. Train model.
    """
    # (1)

    data = load_training_data()

    # (2)

    log('Before preprocessing: ' + str(data))

    data.table = data_preprocessing(data=data.table)
    data.table = get_required_features(data=data.table)

    log('After preprocessing:  ' + str(data))

    # (3)

    random_state = RandomState(seed=42)

    tt_split = data.train_test_split(
        x_cols=config['XCols'],
        y_cols=config['YCols'],
        test_size=0.3,
        shuffle=True,
        random_state=random_state
    )

    # Print out the number of signal and background events in the TT split.

    _train_value_counts = tt_split['YTrain'].value_counts()
    _test_value_counts = tt_split['YTest'].value_counts()

    _train_sig_count = _train_value_counts[1.] #  + _train_value_counts[2.]
    _test_sig_count = _test_value_counts[1.] #  + _test_value_counts[2.]

    _train_bak_count = (
        _train_value_counts[0.]
        # + _train_value_counts[2.]
        + _train_value_counts[3.]
        + _train_value_counts[4.]
    )
    _test_bak_count = (
        _test_value_counts[0.]
        # + _test_value_counts[2.]
        + _test_value_counts[3.]
        + _test_value_counts[4.]
    )

    str_ = ' {:_} signal, {:_} background.'
    log('Train Split:' + str_.format(_train_sig_count, _train_bak_count))
    log('Test Split :' + str_.format(_test_sig_count, _test_bak_count))

    # (4)

    model = RandomForestClassifier()

    log('Model built.')

    # (5)

    log('Training model...')

    model.fit(
        tt_split['XTrain'].to_numpy(),
        tt_split['YTrain'].to_numpy().flatten()
    )

    log('Model trained.')

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

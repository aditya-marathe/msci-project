"""\
src / ana / datasets.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'Datasets'
]

import os

from dotenv import load_dotenv


class Datasets:
    """\
    Datasets
    --------

    Class with attributes storing directories of avalible datasets.
    """
    def __init__(self, verbose: bool = True) -> None:
        """
        `Datasets` constructor.

        Parameters
        ----------
        verbose: bool
            Prints out the found datasets.

        Notes
        -----
        - Must be re-initialised if the `.env` file is updated with new 
          datasets! 
        - Also, when updating the `.env` make sure to follow the convention of
          adding a '_DIR' at the end, otherwise it will not be included!
        """
        load_dotenv()

        ds_names = list()

        for _dataset_name, _dataset_dir in os.environ.items():
            if _dataset_name.endswith('_DIR'):
                setattr(self, _dataset_name, _dataset_dir)
                ds_names.append(_dataset_name)

        if verbose:
            if len(ds_names):
                ds_str = ', '.join(ds_names)
                print(f'Datasets | Found the following: {ds_str}')
            else:
                print('Datasets | No datasets found.')

        del ds_str

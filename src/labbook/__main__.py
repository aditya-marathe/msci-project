"""\
src / labbook (main)
--------------------------------------------------------------------------------

Aditya Marathe
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

import sys

import pathlib

from labbook.browser import LabBookApplication


def main(lb_dir: str | pathlib.Path = './labbook/') -> int:
    """\
    Main
    ----

    Runs the lab book browser to view all the stored models.
    """
    app = LabBookApplication(lb_dir=lb_dir)
    app.mainloop()

    return 0


if __name__ == '__main__':
    sys.exit(main())

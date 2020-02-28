#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    A quick hack of convenience to make it easy to run scripts in an IDE and from command line.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import re
import os


def use_project_path():
    """
    Based on the path of this file, we change directory to the project root.  This is used in the scripts to ensure
    path resolution is done the same when files are run through the IDE and command line.
    :return: the current path, cleaned for URI-based loaders
    """
    path = re.sub(
        '[\\\\/]utility[\\\\/]path\\.py$',
        '',
        __file__
    )
    os.chdir(path)

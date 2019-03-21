
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pydevd


def is_debugging():
    return pydevd.connected


def settrace():
    return pydevd.settrace(suspend=False)

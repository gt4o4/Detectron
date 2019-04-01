
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python import workspace
from six.moves import zip, map
import importlib
import numpy as np
import os


def dump_model(model, blob_list, dump_name, output_dir, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    blob_name = map(core.ScopedName, blob_list)
    data_items = zip(blob_list, workspace.FetchBlobs(blob_name))
    kwargs.update(data_items)
    np.savez_compressed(os.path.join(output_dir, dump_name), **kwargs)


def is_debugging():
    try:
        pydevd = importlib.import_module('pydevd')
    except:
        return False
    return pydevd.connected


def settrace():
    pydevd = importlib.import_module('pydevd')
    return pydevd.settrace(suspend=False)

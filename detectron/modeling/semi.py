
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from caffe2.python import muji
from caffe2.python import core
from caffe2.python import workspace

from detectron.ops.update_semi_labels import UpdateSemiLabelsOp

from detectron.core.config import cfg
import detectron.utils.c2 as c2_utils


def add_semi_update_op(model):
    blobs_in = ['roidb', 'im_info']
    blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
    name = 'UpdateSemiLabelsOp:' + ','.join(
        [str(b) for b in blobs_in]
    )

    # Prepare output blobs
    blobs_out = ['has_label']
    blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

    outputs = model.net.Python(
        UpdateSemiLabelsOp().forward
    )(blobs_in, blobs_out, name=name)

    return outputs

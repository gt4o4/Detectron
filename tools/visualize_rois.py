#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Script for visualizing results saved in a detections.pkl file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys
import numpy as np

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset',
        default='coco_2014_minival',
        type=str
    )
    parser.add_argument(
        '--input-dir',
        dest='input_dir',
        help='input/output directory',
        default='./tmp/vis-output',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--first',
        dest='first',
        help='only visualize the first k images',
        default=0,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def load_dbg(input_dir):
    dbgset = {}
    dbgpath = os.path.join(input_dir, 'dbg')
    for filename in os.listdir(dbgpath):
        npzname, _ = os.path.splitext(filename)
        _, imgname = npzname.split('_', 1)
        dbgset[imgname] = np.load(os.path.join(dbgpath, filename))
    return dbgset


def vis(dataset, input_dir, thresh, limit=0):

    ds = JsonDataset(dataset)
    roidb = ds.get_roidb()

    dets = load_object(os.path.join(input_dir, 'detections.pkl'))

    assert all(k in dets for k in ['all_boxes', 'all_segms', 'all_keyps']), \
        'Expected detections.pkl file in the format used by test_engine.py'

    all_boxes = dets['all_boxes']
    all_segms = dets['all_segms']
    all_keyps = dets['all_keyps']

    def id_or_index(ix, val):
        if len(val) == 0:
            return val
        else:
            return val[ix]

    dbgset = load_dbg(input_dir)

    def fill_rois(img, bbox):
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        h, w, c = img.shape

        if x1 > w:
            x1 = w
        if y1 > h:
            y1 = h

        img[y0:y1, x0:x1, :] = np.maximum(img[y0:y1, x0:x1, :], 1) - 1

        return img

    for ix, entry in enumerate(roidb):
        if limit > 0 and ix >= limit:
            break
        if ix % 10 == 0:
            print('{:d}/{:d}'.format(ix + 1, len(roidb)))

        im = cv2.imread(entry['image'])
        im_name = os.path.splitext(os.path.basename(entry['image']))[0]
        im_dbg = dbgset[im_name]
        im_info = im_dbg['im_info'][0]

        im_w_rois = im[:, :, ::-1]
        roi_boxs = im_dbg['rois'][:, 1:5] / im_info[2]
        for roi in roi_boxs:
            im_w_rois = fill_rois(im_w_rois, roi)
        # for roi in roi_boxs:
        #     im_w_rois = vis_utils.vis_bbox(
        #         fill_rois(im_w_rois, roi), roi,
        #         color=(218, 227, 218))

        cls_boxes_i = [
            id_or_index(ix, cls_k_boxes) for cls_k_boxes in all_boxes
        ]
        cls_segms_i = [
            id_or_index(ix, cls_k_segms) for cls_k_segms in all_segms
        ]
        cls_keyps_i = [
            id_or_index(ix, cls_k_keyps) for cls_k_keyps in all_keyps
        ]

        vis_utils.vis_one_image(
            im_w_rois,
            '{:d}_{:s}'.format(ix, im_name),
            os.path.join(input_dir, 'vis'),
            cls_boxes_i,
            segms=cls_segms_i,
            keypoints=cls_keyps_i,
            thresh=thresh,
            box_alpha=0.8,
            dataset=ds,
            show_class=True
        )


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.dataset,
        opts.input_dir,
        opts.thresh,
        limit=opts.first
    )

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def bbox_transform_inv(boxes, deltas, scale_factors=None):
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if scale_factors:
        dx /= scale_factors[0]
        dy /= scale_factors[1]
        dw /= scale_factors[2]
        dh /= scale_factors[3]

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    predict_xmin = pred_ctr_x - 0.5 * pred_w
    predict_xmax = pred_ctr_x + 0.5 * pred_w
    predict_ymin = pred_ctr_y - 0.5 * pred_h
    predict_ymax = pred_ctr_y + 0.5 * pred_h

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def bbox_transform(ex_rois, gt_rois, scale_factors=None):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    if scale_factors:
        targets_dx *= scale_factors[0]
        targets_dy *= scale_factors[1]
        targets_dw *= scale_factors[2]
        targets_dh *= scale_factors[3]

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets

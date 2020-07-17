# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def focal_loss_(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)

    logits = tf.cast(pred, tf.float32)
    onehot_labels = tf.cast(labels, tf.float32)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    predictions = tf.sigmoid(logits)
    predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t
    positive_mask = tf.cast(tf.greater(labels, 0), tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def focal_loss(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    if anchor_state is not None:
        indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
        labels = tf.gather(labels, indices)
        pred = tf.gather(pred, indices)

    # compute the focal loss
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=pred))
    prediction_probabilities = tf.sigmoid(pred)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (labels * alpha +
                               (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    # compute the normalizer: the number of positive anchors
    if anchor_state is not None:
        normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
        normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
        normalizer = tf.maximum(1.0, normalizer)
    else:
        normalizer = tf.stop_gradient(tf.shape(labels)[0])
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(1.0, normalizer)

    return tf.reduce_sum(focal_cross_entropy_loss) / normalizer


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rcnn(bbox_targets, bbox_pred, anchor_state, sigma=3.0):

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(anchor_state, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, 1, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, 1, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, 1])

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value, 1)*outside_mask) / normalizer

    return bbox_loss


def smooth_l1_loss(targets, preds, anchor_state, sigma=3.0):

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)
    return tf.reduce_sum(regression_loss) / normalizer

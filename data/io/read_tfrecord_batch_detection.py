# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../')

from libs.configs import cfgs
from data.io import image_preprocess_batch

def parse_record(raw_record):
    """Parse data from a tf record."""
    keys_to_features = {
        'img_name': tf.FixedLenFeature([], tf.string),
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'img': tf.FixedLenFeature([], tf.string),
        'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
    }
    # 读入一个样例
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    
    img_name  = parsed['img_name']
    img_height = tf.cast(parsed['h'], tf.int32)
    img_width = tf.cast(parsed['w'], tf.int32)

    img = tf.decode_raw(parsed['img'], tf.uint8)
    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(parsed['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
    
    img, gtboxes_and_label, h, w = preprocess_img(img, gtboxes_and_label)#,is_training)

    num_obs = tf.cast(tf.shape(gtboxes_and_label)[0],tf.int32)

    return img, gtboxes_and_label, num_obs, h, w

def preprocess_img(img, gtbox, is_training=True):
    img = tf.cast(img, tf.float32)
    
    img, gtboxes_and_label, h, w= image_preprocess_batch.short_side_resize(img_tensor=img,
                                                                gtboxes_and_label=gtbox,
                                                                target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                                length_limitation=cfgs.IMG_MAX_LENGTH)
    if is_training:
        img, gtboxes_and_label = image_preprocess_batch.random_flip_left_right(img_tensor=img,
                                                                        gtboxes_and_label=gtboxes_and_label)
    # gtboxes_and_label=gtbox
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img =(img / 255 - tf.constant([[cfgs.PIXEL_MEAN_]]))/tf.constant([[cfgs.PIXEL_STD]])
    else:
        img = img - tf.constant([[cfgs.PIXEL_MEAN]])  # sub pixel mean at last

    return img, gtboxes_and_label, h, w


def next_batch(batch_size,is_training=True):
    ''' 
    img_tensor:[h, w, c], 
    gtboxes_and_label:[-1, 5]:[x1, y1, x2, y2, label]
    '''
    #assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if is_training:
        name='train.tfrecord'
    else:
        name='test.tfrecord'

    pattern = os.path.join('/data/scene/', name)
    # pattern = os.path.join('../VGDataset/', name)
    print('tfrecord path is -->', os.path.abspath(pattern))

    dataset = tf.data.Dataset.from_tensor_slices([pattern])
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(parse_record)
    dataset = dataset.prefetch(buffer_size=10 * batch_size)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=100 * batch_size)

    if is_training:
        dataset = dataset.repeat()

    dataset=dataset.padded_batch(batch_size,padded_shapes=([None,None,None],[None,None],[],[],[]))

    # if is_training:
    #     dataset = dataset.repeat()


    iterator = dataset.make_one_shot_iterator()
    img, gtboxes_and_label, num_obs, h, w = iterator.get_next()

    return img, gtboxes_and_label, num_obs, h, w


if __name__ == '__main__':
    import cv2
    import time

    img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch =next_batch(2,is_training=True)
    print(num_objects_batch.shape, img_h_batch.shape, img_w_batch.shape)
    inputs_list = []
    for i in range(3):
        num_objects = num_objects_batch[i]
        img_h = img_h_batch[i]
        img_w = img_w_batch[i]
        img=tf.reshape(img_batch[i,:img_h,:img_w,:],(img_h,img_w,3))
        gtboxes_and_label = tf.cast(tf.reshape(gtboxes_and_label_batch[i,:num_objects,:], [-1, 5]), tf.float32)
        inputs_list.append([img, gtboxes_and_label])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(2):
            
            img,gtboxes_and_label=inputs_list[i]
            start = time.time()
            i,gt=sess.run([img,gtboxes_and_label])
            end = time.time()
            print(i.shape)
            print(gt.shape)

            i = (i*np.array(cfgs.PIXEL_STD)+ np.array(cfgs.PIXEL_MEAN_)) * 255
            i = np.array(i * 255 / np.max(i), dtype=np.uint8)

            rel_img=i[:,:,::-1].copy()
            for idx in range(gt.shape[0]):
                x1, y1, x2, y2, label=gt[idx]
                cv2.rectangle(rel_img,(x1, y1),(x2, y2),(0,0,255),2,0)
            cv2.imshow('rel_img',rel_img)
            cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads)

# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.networks.build_whole_network import DetectionNetwork
from libs.detection_oprations.proposal_rel_opr import assign_tensor
from libs.detection_oprations.proposal_rel_opr import postprocess_detctions
from libs.detection_oprations.proposal_rel_opr import full_connection_graph_eval_predcls

from libs.detection_oprations.proposal_rel_opr import full_connection_graph
from libs.detection_oprations.proposal_rel_opr import full_connection_graph_inference

class RelDetectionNetwork(object):

    def __init__(self, base_network_name,mode):

        self.base_network_name = base_network_name
        self.mode=mode
        self.is_training = False
        self.det_net=DetectionNetwork(self.base_network_name,False,True)

    def assign_levels(self, all_rois, scope='rel'):
        '''
        :param all_rois:
        :return:
        '''
        with tf.name_scope('assign_'+scope+'_levels'):
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)
            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)

            # use floor instead of round; 4 + log_2(***)
            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  

            min_level = int(cfgs.LEVEL[0][-1])
            max_level = min(6, int(cfgs.LEVEL[-1][-1]))
            # level minimum is 3
            levels = tf.maximum(levels, tf.ones_like(levels) * min_level)
            # level maximum is 6
            levels = tf.minimum(levels, tf.ones_like(levels) * max_level)
            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            def summary_rois(levels, level_i):
                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])
                tf.summary.scalar('LEVEL/'+scope+'_LEVEL_%d_rois_NUM'%level_i, tf.shape(level_i_indices)[0])

            for i in range(min_level,max_level+1):
                summary_rois(levels=levels, level_i=i)

            return levels # 3-6, [P3, P4, P5, P6] Note: P7 do not assign rois

    def roi_pooling(self, feature_maps, rois, img_shape, scope):
        '''
        :param feature_maps: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''
        with tf.variable_scope('ROI_Warping_'+scope):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(image=feature_maps, 
                                                            boxes=normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ], dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE')

            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE,
                                           padding='SAME')

        return roi_features

    def assign_features(self, feature_pyramid, levels, boxes, img_shape):
        min_level = int(cfgs.LEVEL[0][-1])
        max_level = min(6, int(cfgs.LEVEL[-1][-1]))

        pool_features=tf.zeros([tf.shape(boxes)[0],int(cfgs.ROI_SIZE/cfgs.ROI_POOL_KERNEL_SIZE),int(cfgs.ROI_SIZE/cfgs.ROI_POOL_KERNEL_SIZE),cfgs.CHANNELS_FEATURE])
        with tf.variable_scope('rois_pooling'):
            for level_name in cfgs.LEVEL:
                cur_level=min(max(int(level_name[-1]),min_level),max_level)
                level_i_indices=tf.where(tf.equal(levels,cur_level))
                cur_boxes=tf.reshape(tf.gather(boxes,level_i_indices),[-1,4])
                rois_features=self.roi_pooling(feature_pyramid[level_name],cur_boxes,img_shape,level_name)
                pool_features=tf.py_func(func=assign_tensor, inp=[pool_features, tf.squeeze(level_i_indices), rois_features], Tout=[tf.float32])
                pool_features=tf.reshape(pool_features,[tf.shape(boxes)[0],int(cfgs.ROI_SIZE/cfgs.ROI_POOL_KERNEL_SIZE),int(cfgs.ROI_SIZE/cfgs.ROI_POOL_KERNEL_SIZE),cfgs.CHANNELS_FEATURE])

        return pool_features             

    def feature_attention(self,sub_feature,obj_feature,rel_feature):
        # subject feature 
        sub_feature_1=sub_feature
        for i in range(4):
            sub_feature_1= slim.conv2d(inputs=sub_feature_1,
                                       num_outputs=256,
                                       kernel_size=[3,3],
                                       stride=1,
                                       activation_fn=None,#tf.nn.relu if i<3 else None,
                                       weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                       biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                       scope='sub_feature_1_{}'.format(i),
                                       trainable=self.is_training)
            #!!!!!!!!!!!!!
            sub_feature_1= slim.batch_norm(sub_feature_1)#,is_training=self.is_training)
            if i<3:
                sub_feature_1=tf.nn.relu(sub_feature_1)

        sub_feature_2=sub_feature
        for i in range(4):                  
            sub_feature_2= slim.conv2d(inputs=sub_feature_2,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,#tf.nn.relu if i<3 else None,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='sub_feature_2_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            sub_feature_2= slim.batch_norm(sub_feature_2)#,is_training=self.is_training)
            if i<3:
                sub_feature_2=tf.nn.relu(sub_feature_2)
                
        # object feature
        obj_feature_1=obj_feature
        for i in range(4):
            obj_feature_1= slim.conv2d(inputs=obj_feature_1,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,#tf.nn.relu if i<3 else None,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='obj_feature_1_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            obj_feature_1= slim.batch_norm(obj_feature_1)#,is_training=self.is_training)
            if i<3:
                obj_feature_1=tf.nn.relu(obj_feature_1)

        obj_feature_2=obj_feature
        for i in range(4):
            obj_feature_2= slim.conv2d(inputs=obj_feature_2,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=tf.nn.relu if i<3 else None,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='obj_feature_2_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            obj_feature_2= slim.batch_norm(obj_feature_2)#,is_training=self.is_training)
            if i<3:
                obj_feature_2=tf.nn.relu(obj_feature_2)

        sub_obj_featur_1 = tf.nn.relu(tf.add(sub_feature_1, obj_feature_1))
        sub_obj_featur_2 = tf.nn.relu(tf.add(sub_feature_2, obj_feature_2))

        # relation feature
        for i in range(4):
            rel_feature= slim.conv2d(inputs=rel_feature,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,#tf.nn.relu if i<3 else None,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='rel_feature_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            rel_feature= slim.batch_norm(rel_feature)#,is_training=self.is_training)
            if i<3:
                rel_feature=tf.nn.relu(rel_feature)

        # sub_obj_featur_1 = tf.add(sub_feature_1, obj_feature_1)        
        # attention_feature =tf.nn.relu(tf.add(sub_feature_1, rel_feature))

        attention_feature=tf.multiply(rel_feature,sub_obj_featur_1)
        attention_feature=tf.reshape( tf.nn.softmax( tf.reshape(attention_feature,[tf.shape(attention_feature)[0],-1,tf.shape(attention_feature)[-1]]), axis=1 ), tf.shape(attention_feature) )
        rel_feature= tf.multiply(attention_feature, rel_feature)
        attention_feature=tf.add(rel_feature,sub_obj_featur_2)

        for i in range(4):
            attention_feature= slim.conv2d(inputs=attention_feature,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,#tf.nn.relu,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='attention_feature_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            attention_feature= slim.batch_norm(attention_feature)#,is_training=self.is_training)
            attention_feature=tf.nn.relu(attention_feature)

        return attention_feature

    def attention_refine(self,feature):
        spatial_attention=slim.conv2d(inputs=feature,
                                    num_outputs=1,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='spatial_attention',
                                    trainable=self.is_training)
        #!!!!!!!!!!!!!        
        spatial_attention= slim.batch_norm(spatial_attention)#,is_training=self.is_training)
        spatial_attention=tf.reshape(tf.nn.softmax(tf.reshape(spatial_attention,[tf.shape(spatial_attention)[0],-1,tf.shape(spatial_attention)[-1]]),axis=1),tf.shape(spatial_attention))
        spatial_attention=tf.multiply(spatial_attention,feature)

        channel_attention=slim.avg_pool2d(inputs=feature,
                                          kernel_size=[cfgs.ROI_SIZE//cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_SIZE//cfgs.ROI_POOL_KERNEL_SIZE],
                                          stride=1)
        for i in range(2):
            channel_attention=slim.conv2d(inputs=channel_attention,
                                        num_outputs=256,
                                        kernel_size=[1,1],
                                        stride=1,
                                        activation_fn=None,#tf.nn.relu,
                                        weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                        biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                        scope='channel_attention_{}'.format(i),
                                        trainable=self.is_training)
            #!!!!!!!!!!!!!
            channel_attention= slim.batch_norm(channel_attention)#,is_training=self.is_training)
            if i<1:
                channel_attention= tf.nn.relu(channel_attention)
        channel_attention=tf.nn.sigmoid(channel_attention)

        refine_feature= tf.multiply(spatial_attention,channel_attention)
        for i in range(2):
            refine_feature= slim.conv2d(inputs=refine_feature,
                                    num_outputs=256,
                                    kernel_size=[3,3],
                                    stride=1,
                                    activation_fn=None,#tf.nn.relu,
                                    weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                    biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                    scope='refine_feature_{}'.format(i),
                                    trainable=self.is_training)
            #!!!!!!!!!!!!!
            refine_feature= slim.batch_norm(refine_feature)#,is_training=self.is_training)
            refine_feature=tf.nn.relu(refine_feature)

        return refine_feature      

    def build_whole_rel_network(self, input_img_batch, R_matrix, gt_boxes):
        '''
        R_matrix: n x n
        gt_boxes: n x 4
        '''
        img_shape = tf.shape(input_img_batch)
        if self.mode == 'predcls':
            gt_boxes = tf.reshape(gt_boxes, [-1, 4])
            gt_boxes = tf.cast(gt_boxes, tf.float32)
        # else:# self.mode == 'sggen':
        #     gt_boxes = tf.reshape(gt_boxes, [-1, 5])
        #     gt_boxes = tf.cast(gt_boxes, tf.float32)

        # 1. detection network
        boxes,scores,category,feature_pyramid = self.det_net.build_whole_detection_network(input_img_batch=input_img_batch,gtboxes_batch=None)

        # 2. union boxes, subjec/object boxes, rel_labels
        with tf.variable_scope('build_union_boxes'):
            if self.mode == 'predcls':
                # [pos+neg,4],[pos+neg,4],[pos+neg,4],[pos+neg,]
                sub_boxes,obj_boxes,rel_boxes,rel_labels = tf.py_func(func=full_connection_graph_eval_predcls,
                                                                      inp=[R_matrix, gt_boxes],
                                                                      Tout=[tf.float32, tf.float32, tf.float32, tf.int32])
            else:# self.mode == 'sggen':
                sub_boxes,obj_boxes,sub_categorys,obj_categorys,sub_scores,obj_scores,rel_boxes = \
                    tf.py_func(func=full_connection_graph_inference,
                               inp=[boxes, scores, category],
                               Tout=[tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32, tf.float32])

        sub_boxes=tf.reshape(sub_boxes,[-1,4])
        obj_boxes=tf.reshape(obj_boxes,[-1,4])
        rel_boxes=tf.reshape(rel_boxes,[-1,4])

        # 3. assign boxes
        sub_levels=self.assign_levels(sub_boxes,scope='sub')
        obj_levels=self.assign_levels(obj_boxes,scope='obj')
        rel_levels=self.assign_levels(rel_boxes,scope='rel')

        # 4. assign roi's feature
        sub_features=self.assign_features(feature_pyramid,sub_levels,sub_boxes,img_shape)
        obj_features=self.assign_features(feature_pyramid,obj_levels,obj_boxes,img_shape)
        rel_features=self.assign_features(feature_pyramid,rel_levels,rel_boxes,img_shape)

        with tf.variable_scope('build_attention', regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
            # 5. feature attention
            attention_feature=self.feature_attention(sub_features,obj_features,rel_features)
            # 6. attention refine
            refine_feature=self.attention_refine(attention_feature)
            # 7. fc
            flatten_feature = slim.flatten(inputs=refine_feature, scope='flatten_inputs')
            flatten_feature = slim.fully_connected(inputs=flatten_feature,
                                                   num_outputs=1024,
                                                   weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                                   biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                                   trainable=self.is_training,
                                                   scope='fc1')
            flatten_feature = slim.fully_connected(inputs=flatten_feature,
                                                   num_outputs=1024,
                                                   weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                                   biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                                   trainable=self.is_training,
                                                   scope='fc2')
            rel_pred = slim.fully_connected(inputs=flatten_feature,
                                            num_outputs=cfgs.REL_CLASS_NUM,
                                            weights_initializer=cfgs.SUBNETS_WEIGHTS_INITIALIZER,
                                            biases_initializer=cfgs.SUBNETS_BIAS_INITIALIZER,
                                            activation_fn=None, 
                                            trainable=self.is_training,
                                            scope='rel_fc')

            rel_prob=tf.nn.sigmoid(rel_pred)
            rel_preds=tf.argmax(rel_prob,axis=1)+1
            rel_prob=tf.reduce_max(rel_prob,axis=1)
        
        if self.mode == 'predcls':
            return rel_labels,rel_preds,rel_prob
        else:# self.mode == 'sggen':
            return rel_prob,rel_preds,sub_boxes,obj_boxes,sub_categorys,obj_categorys
            
    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.REL_VERSION))
        restorer = None
        det_checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.DETECTION_VERSION))
        det_restorer = None
        if checkpoint_path != None:
            model_variables = slim.get_model_variables()
            restore_variables = [var for var in model_variables if var.name.startswith('build_attention')] + \
                                [slim.get_or_create_global_step()]
            restorer = tf.train.Saver(restore_variables)
            print("rel model restore from :", checkpoint_path)

        if det_checkpoint_path != None:
            model_variables = slim.get_model_variables()
            restore_variables = [var for var in model_variables if not var.name.startswith('build_attention')] 
            det_restorer = tf.train.Saver(restore_variables)
            print("det model restore from :", det_checkpoint_path)

        return restorer, checkpoint_path, det_restorer, det_checkpoint_path

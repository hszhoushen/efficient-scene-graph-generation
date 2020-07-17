# encoding: utf-8
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../')

from libs.configs import cfgs
from libs.box_utils import boxes_utils
from libs.box_utils import bbox_transform

def full_connection_graph(R_matrix, obj_boxes):
    '''
    R_matrix: n x n
    obj_boxes: n x 4
    '''
    pos=np.where(R_matrix>0)
    neg=np.where(R_matrix==0)

    pos_rel=R_matrix[pos]
    pos_sub_boxes=obj_boxes[pos[0]]
    pos_obj_boxes=obj_boxes[pos[1]]

    neg_sub_boxes=obj_boxes[neg[0]]
    neg_obj_boxes=obj_boxes[neg[1]]

    num_pos=pos_sub_boxes.shape[0]
    num_neg=neg_sub_boxes.shape[0]
    
    if num_pos>cfgs.REL_BATCH_SIZE*cfgs.REL_FRACTION:
        sample=np.random.choice(np.arange(pos_sub_boxes.shape[0]),int(cfgs.REL_BATCH_SIZE*cfgs.REL_FRACTION),replace=False)
        pos_rel=pos_rel[sample]
        pos_sub_boxes=pos_sub_boxes[sample]
        pos_obj_boxes=pos_obj_boxes[sample]
    
    if num_neg>cfgs.REL_BATCH_SIZE*(1-cfgs.REL_FRACTION):
        sample=np.random.choice(np.arange(neg_sub_boxes.shape[0]),int(cfgs.REL_BATCH_SIZE*(1-cfgs.REL_FRACTION)),replace=False)
        neg_sub_boxes=neg_sub_boxes[sample]
        neg_obj_boxes=neg_obj_boxes[sample]
    
    if num_pos<num_neg:
        sample=np.random.choice(np.arange(neg_sub_boxes.shape[0]),num_pos,replace=False)
        neg_sub_boxes=neg_sub_boxes[sample]
        neg_obj_boxes=neg_obj_boxes[sample]
    
    sub_boxes=np.concatenate([pos_sub_boxes,neg_sub_boxes],axis=0)
    obj_boxes=np.concatenate([pos_obj_boxes,neg_obj_boxes],axis=0)
    neg_rel=np.zeros(neg_sub_boxes.shape[0],dtype=np.int32)
    rel=np.concatenate([pos_rel,neg_rel],axis=0)

    rel_labels=np.zeros((rel.shape[0],cfgs.REL_CLASS_NUM),dtype=np.float32)
    for i in range(rel.shape[0]):
        if rel[i]!=0:
            rel_labels[i,rel[i]-1]=1

    x1=np.minimum(sub_boxes[:,0],obj_boxes[:,0])
    y1=np.minimum(sub_boxes[:,1],obj_boxes[:,1])
    x2=np.maximum(sub_boxes[:,2],obj_boxes[:,2])
    y2=np.maximum(sub_boxes[:,3],obj_boxes[:,3])
    rel_boxes=np.transpose(np.stack([x1,y1,x2,y2],axis=0))

    return sub_boxes,obj_boxes,rel_boxes,rel_labels

def full_connection_graph_eval_predcls(R_matrix, obj_boxes):
    '''
    R_matrix: n x n
    obj_boxes: n x 4
    '''
    pos=np.where(R_matrix>0)
    neg=np.where(R_matrix==0)

    pos_rel=R_matrix[pos]
    pos_sub_boxes=obj_boxes[pos[0]]
    pos_obj_boxes=obj_boxes[pos[1]]

    neg_sub_boxes=obj_boxes[neg[0]]
    neg_obj_boxes=obj_boxes[neg[1]]

    num_pos=pos_sub_boxes.shape[0]
    num_neg=neg_sub_boxes.shape[0]
    
    sub_boxes=np.concatenate([pos_sub_boxes,neg_sub_boxes],axis=0)
    obj_boxes=np.concatenate([pos_obj_boxes,neg_obj_boxes],axis=0)
    neg_rel=np.zeros(neg_sub_boxes.shape[0],dtype=np.int32)
    rel=np.concatenate([pos_rel,neg_rel],axis=0)

    x1=np.minimum(sub_boxes[:,0],obj_boxes[:,0])
    y1=np.minimum(sub_boxes[:,1],obj_boxes[:,1])
    x2=np.maximum(sub_boxes[:,2],obj_boxes[:,2])
    y2=np.maximum(sub_boxes[:,3],obj_boxes[:,3])
    rel_boxes=np.transpose(np.stack([x1,y1,x2,y2],axis=0))

    if rel_boxes.shape[0]>1024:
        sample=np.random.choice(np.arange(rel_boxes.shape[0]),1024,replace=False)
        sub_boxes,obj_boxes,rel_boxes,rel=sub_boxes[sample],obj_boxes[sample],rel_boxes[sample],rel[sample]

    return sub_boxes,obj_boxes,rel_boxes,rel


def full_connection_graph_inference(boxes, scores, categorys):
    '''
    obj_boxes: [n, 4]
    obj_scores: [n,]
    obj_categorys: [n,]
    '''
    sub_boxes,obj_boxes,sub_categorys,obj_categorys,sub_scores,obj_scores=[],[],[],[],[],[]
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            if i==j:
                continue 
            sub_boxes.append(boxes[i])
            sub_scores.append(scores[i])            
            sub_categorys.append(categorys[i])

            obj_boxes.append(boxes[j])
            obj_scores.append(scores[j])            
            obj_categorys.append(categorys[j])

    sub_boxes=np.array(sub_boxes,dtype=np.float32)
    sub_scores=np.array(sub_scores,dtype=np.float32)
    sub_categorys=np.array(sub_categorys,dtype=np.int32)

    obj_boxes=np.array(obj_boxes,dtype=np.float32)
    obj_scores=np.array(obj_scores,dtype=np.float32)
    obj_categorys=np.array(obj_categorys,dtype=np.int32)

    if sub_boxes.shape[0]>0:
        x1=np.minimum(sub_boxes[:,0],obj_boxes[:,0])
        y1=np.minimum(sub_boxes[:,1],obj_boxes[:,1])
        x2=np.maximum(sub_boxes[:,2],obj_boxes[:,2])
        y2=np.maximum(sub_boxes[:,3],obj_boxes[:,3])
        rel_boxes=np.transpose(np.stack([x1,y1,x2,y2],axis=0))
    else:
        rel_boxes=np.zeros((0,4),dtype=np.float32)

    return sub_boxes,obj_boxes,sub_categorys,obj_categorys,sub_scores,obj_scores,rel_boxes

def filter_detections(scores, is_training):
    """
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.REL_VIS_SCORE)), [-1, ])
    else:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.REL_FILTERED_SCORE)), [-1, ])
    # add indices to list of all indices
    return indices

def postprocess_detctions(rel_cls_prob, rel_boxes, sub_boxes, obj_boxes, sub_categorys, obj_categorys, sub_scores, obj_scores, is_training):
    return_rel_scores = []
    return_rel_labels = []
    return_rel_boxes = []
        
    return_obj_boxes = []
    return_sub_boxes = []

    if not is_training:
        return_sub_scores = []
        return_sub_labels = []
        return_obj_scores = []
        return_obj_labels = []
    
    for j in range(0, cfgs.REL_CLASS_NUM):
        indices = filter_detections(rel_cls_prob[:, j], is_training)

        tmp_rel_boxes = tf.reshape(tf.gather(rel_boxes, indices), [-1, 4])
        tmp_rel_scores = tf.reshape(tf.gather(rel_cls_prob[:, j], indices), [-1, ])
        tmp_rel_labels = tf.ones_like(tmp_rel_scores)*(j+1)

        return_rel_boxes.append(tmp_rel_boxes)
        return_rel_scores.append(tmp_rel_scores)
        return_rel_labels.append(tmp_rel_labels)

        tmp_sub_boxes = tf.reshape(tf.gather(sub_boxes, indices), [-1, 4])
        tmp_obj_boxes = tf.reshape(tf.gather(obj_boxes, indices), [-1, 4])
        return_obj_boxes.append(tmp_obj_boxes)
        return_sub_boxes.append(tmp_sub_boxes)
        
        if not is_training:
            tmp_sub_scores = tf.reshape(tf.gather(sub_scores, indices), [-1, ])
            tmp_sub_labels = tf.reshape(tf.gather(sub_categorys, indices), [-1, ])
            tmp_obj_scores = tf.reshape(tf.gather(obj_scores, indices), [-1, ])
            tmp_obj_labels = tf.reshape(tf.gather(obj_categorys, indices), [-1, ])

            return_sub_scores.append(tmp_sub_scores)
            return_sub_labels.append(tmp_sub_labels)    
            return_obj_scores.append(tmp_obj_scores)
            return_obj_labels.append(tmp_obj_labels)

    return_rel_boxes = tf.concat(return_rel_boxes, axis=0)
    return_rel_scores = tf.concat(return_rel_scores, axis=0)
    return_rel_labels = tf.concat(return_rel_labels, axis=0)

    return_sub_boxes = tf.concat(return_sub_boxes, axis=0)
    return_obj_boxes = tf.concat(return_obj_boxes, axis=0)

    if not is_training:
        return_sub_scores = tf.concat(return_sub_scores, axis=0)
        return_sub_labels = tf.concat(return_sub_labels, axis=0)

        return_obj_scores = tf.concat(return_obj_scores, axis=0)
        return_obj_labels = tf.concat(return_obj_labels, axis=0)

    if is_training:
        return return_rel_boxes, return_rel_scores, return_rel_labels, return_sub_boxes, return_obj_boxes
    else:
        return return_rel_boxes, return_rel_scores, return_rel_labels, return_sub_boxes, return_sub_scores, return_sub_labels, return_obj_boxes, return_obj_scores, return_obj_labels

def assign_tensor(var,indice,sub_var):
    var[indice]=sub_var
    return var

if __name__ == '__main__':
    # R=np.array([[-1,10,20,30],[0,-1,0,15],[0,40,-1,0],[0,0,0,-1]])
    # obj_boxes=np.array([[10,20,30,40],[25,30,40,55],[5,30,20,80],[50,60,70,80]])
    # sub_boxes,obj_boxes,rel_boxes,rel_labels=full_connection_graph(R,obj_boxes)
    
    # print(sub_boxes)
    # print('-'*30)
    # print(obj_boxes)
    # print('-'*30)
    # print(rel_boxes)
    # print('-'*30)
    # print(rel_labels)

    obj_boxes=np.array([[10,20,30,40],[25,30,40,55],[5,30,20,80],[50,60,70,80]])
    obj_scores=np.array([0.8,0.5,0.4,0.3])
    obj_categorys=np.array([10,20,40,60])

    sub_boxes,obj_boxes,sub_categorys,obj_categorys,sub_scores,obj_scores,rel_boxes = full_connection_graph_inference(obj_boxes,obj_scores,obj_categorys)
    
    print(sub_boxes)
    print('-'*30)
    print(obj_boxes)
    print('-'*30)
    print(rel_boxes)
    print('-'*30)
    print(sub_categorys)
    print('-'*30)
    print(obj_categorys)


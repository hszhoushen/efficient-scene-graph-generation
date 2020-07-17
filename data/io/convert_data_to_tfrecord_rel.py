# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')

import os
import cv2
import json
import numpy as np
import tensorflow as tf

from help_utils.tools import *

from libs.configs import cfgs
from libs.label_name_dict.label_dict import *

# tf.app.flags.DEFINE_string('base_dir', '../VGDataset/', 'Base dir')
tf.app.flags.DEFINE_string('base_dir', '/data/scene/', 'Base dir')
tf.app.flags.DEFINE_string('img_dir1', 'images/VG_100K', 'img dir1')
tf.app.flags.DEFINE_string('img_dir2','images2/VG_100K_2', 'img dir2')

tf.app.flags.DEFINE_string('object_list_dir', '../VGDataset/VG_list/object_list.txt', 'object list dir')
tf.app.flags.DEFINE_string('object_alias_dir','../VGDataset/VG_list/object_alias.txt','object alias dir')
tf.app.flags.DEFINE_string('predicate_list_dir', '../VGDataset/VG_list/predicate_list.txt', 'predicate list dir')
tf.app.flags.DEFINE_string('predicate_alias_dir','../VGDataset/VG_list/predicate_alias.txt', 'predicate alias dir')

tf.app.flags.DEFINE_string('label_file', 'scene_graphs.json', 'label file dir')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')

tf.app.flags.DEFINE_string('train_rpn_save_name', 'train_rel', 'train rpn save name')
tf.app.flags.DEFINE_string('test_rpn_save_name' , 'test_rel' , 'test rpn save name' )
FLAGS = tf.app.flags.FLAGS

STATISTICS_OBJ_DICT={}
STATISTICS_REL_DICT={}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab

def get_label(info,obj_alias_dict,rel_alias_dict):
    relationships=info['relationships']
    image_id=info['image_id']
    objects=info['objects']

    ob_dict=dict()
    gt_boxes=[]
    for idx,ob in enumerate(objects):
        ob_id=ob['object_id']
        ob_name=ob['names'][0]
        ob_x,ob_y,ob_w,ob_h=ob['x'],ob['y'],ob['w'],ob['h']
        ob_dict[ob_id]={'box':[ob_x,ob_y,ob_x+ob_w,ob_y+ob_h],'name':ob_name,'r_id':idx}
        gt_boxes.append([ob_x,ob_y,ob_x+ob_w,ob_y+ob_h,-1])

    R_matrix=np.zeros([len(objects),len(objects)],dtype=np.int32)
    for i,j in zip(range(len(objects)),range(len(objects))):
        R_matrix[i,j]=-1

    for re in relationships:
        re_oid=re['object_id']
        re_sid=re['subject_id']
        re_predicate=re['predicate']

        ob_name=ob_dict[re_oid]['name']
        sb_name=ob_dict[re_sid]['name']
        ob_index=ob_dict[re_oid]['r_id']
        sb_index=ob_dict[re_sid]['r_id']

        if ob_name in obj_alias_dict:
            ob_name=obj_alias_dict[ob_name]
        if sb_name in obj_alias_dict:
            sb_name=obj_alias_dict[sb_name]
        if re_predicate in rel_alias_dict:
            re_predicate=rel_alias_dict[re_predicate]

        if ob_name in obj_classes_ID:
            if ob_name in STATISTICS_OBJ_DICT:
                STATISTICS_OBJ_DICT[ob_name]+=1
            else:
                STATISTICS_OBJ_DICT[ob_name]=1
            ob_id=obj_classes_ID[ob_name]
        else:
            continue # 跳过obj为other的

        if sb_name in obj_classes_ID:
            if sb_name in STATISTICS_OBJ_DICT:
                STATISTICS_OBJ_DICT[sb_name]+=1
            else:
                STATISTICS_OBJ_DICT[sb_name]=1
            sb_id=obj_classes_ID[sb_name]
        else:
            continue # 跳过obj为other的

        if re_predicate in rel_classes_ID:
            if re_predicate in STATISTICS_REL_DICT:
                STATISTICS_REL_DICT[re_predicate]+=1
            else:
                STATISTICS_REL_DICT[re_predicate]=1
            re_id=rel_classes_ID[re_predicate]
        else:
            continue # 跳过relationship为other的
        
        gt_boxes[sb_index][-1]=sb_id
        gt_boxes[ob_index][-1]=ob_id
        R_matrix[sb_index,ob_index]=re_id

    return R_matrix,np.array(gt_boxes,dtype=np.int32)

def convert_vg_to_tfrecord():
    json_path = FLAGS.base_dir + FLAGS.label_file
    image1_path = FLAGS.base_dir + FLAGS.img_dir1
    image2_path = FLAGS.base_dir + FLAGS.img_dir2

    train_save_path = FLAGS.base_dir + FLAGS.train_rpn_save_name + '.tfrecord'
    test_save_path  = FLAGS.base_dir + FLAGS.test_rpn_save_name  + '.tfrecord'
    
    obj_alias_dict, all_obj_list = make_alias_dict(FLAGS.object_alias_dir)
    rel_alias_dict, all_rel_list = make_alias_dict(FLAGS.predicate_alias_dir)

    writer = tf.python_io.TFRecordWriter(path=train_save_path)
    test_writer = tf.python_io.TFRecordWriter(path=test_save_path)

    f = open(json_path, encoding='utf-8')  
    f = json.load(f)

    images1=dict()
    images_list=os.listdir(image1_path)
    for img_name in images_list:
        img_id=int(img_name.split('.')[0])
        images1[img_id]=1
    
    images2=dict()
    images_list=os.listdir(image2_path)
    for img_name in images_list:
        img_id=int(img_name.split('.')[0])
        images2[img_id]=1

    trian_count,test_count=0,0
    for count,info in enumerate(f):
        view_bar('Conversion progress', count + 1, len(f))

        img_id=info['image_id']
        if img_id in images1:
            img_path = os.path.join(image1_path,str(img_id)+'.jpg')
        elif img_id in images2:
            img_path = os.path.join(image2_path,str(img_id)+'.jpg')
        else:
            continue

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img=cv2.imread(img_path)[:, :, ::-1] # BGR->RGB
        R_matrix,gt_boxes=get_label(info,obj_alias_dict,rel_alias_dict)
        if gt_boxes.shape[0]<=0 or np.sum(R_matrix>0)<=0:
            continue

        feature = tf.train.Features(feature={
            'img_name': _bytes_feature((str(img_id)+'.jpg').encode()),
            'img': _bytes_feature(img.tostring()),
            'h': _int64_feature(img.shape[0]),
            'w': _int64_feature(img.shape[1]),
            'r_matrix': _bytes_feature(R_matrix.tostring()),
            'gt_boxes': _bytes_feature(gt_boxes.tostring())
        })

        example = tf.train.Example(features=feature)

        prob=np.random.random()
        if prob<=0.8:
            writer.write(example.SerializeToString())
            trian_count+=1
        else:
            test_writer.write(example.SerializeToString())
            test_count+=1

    print('\nConversion is complete!')
    
    with open('../VGDataset/data_rel_statistics.txt','w') as f:
        f.write('1. COUNT:\n')
        f.write('   train count:'+str(trian_count)+'\n')
        f.write('   test count:'+str(test_count)+'\n')
        f.write('\n')
        f.write('2. OBJECTS:\n')
        for key in STATISTICS_OBJ_DICT:
            f.write('   '+key+':'+str(STATISTICS_OBJ_DICT[key])+'\n')
        f.write('\n')
        f.write('3. PREDICATES:\n')
        for key in STATISTICS_REL_DICT:
            f.write('   '+key+':'+str(STATISTICS_REL_DICT[key])+'\n')

if __name__ == '__main__':
    convert_vg_to_tfrecord()

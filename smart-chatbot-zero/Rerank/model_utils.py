# -*- coding: utf-8 -*-
# @Time    : 2019/5/24 20:04
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model_utils.py
# @Software: PyCharm


import tensorflow as tf
import numpy as np




# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    try:
        print('{0} : {1}'.format(varname, var.get_shape()))
    except:
        print('{0} : {1}'.format(varname, np.shape(var)))


# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


# 余弦相似度计算
def feature2cos_sim(feat_q, feat_a):
    # feat_q: 2D:(bz, hz)
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return tf.clip_by_value(cos_sim_q_a, 1e-5, 0.99999)
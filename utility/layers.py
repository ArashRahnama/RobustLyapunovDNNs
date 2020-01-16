#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:23:11 2018
by Arash Rahnama

"""
import tensorflow as tf
from operator import mul
from functools import reduce

from utility import utils as u

def linear(x, output_size, scope_name='linear', spectral_norm=True, update_collection=None, 
           l2_norm=False, weight_decay=0, xavier=True, rho=1.0, reuse=None):
    # a linear fully connected layer with spectral normalization and weight decay options
    shape = x.get_shape().as_list()

    if len(shape) > 2:
        x_flat = tf.reshape(x, [-1, reduce(mul, shape[1:])])
    else:
        x_flat = x
        
    shape = x_flat.get_shape()
    input_size = shape[1]
    
    with tf.variable_scope(scope_name, reuse=reuse):
        
        if xavier:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
            
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        
        if spectral_norm:
            weights = u.weights_spectral_norm_linear(weights, update_collection=update_collection, gamma=rho)
        elif l2_norm:
            wd = tf.multiply(tf.nn.l2_loss(weights), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', wd)
        
        output = tf.matmul(x_flat, weights) + bias
        
        return output

def conv2d(x, kernel_size, scope_name='conv2d', stride=1, padding='SAME', use_bias=True, 
           spectral_norm=True, update_collection=None, l2_norm=False, weight_decay=0, variance_scaling=False,
           xavier=True, batch_normal = False, rho=1.0, training=False, reuse=None):
    # a 2D convolution layer with spectral normalization and weight decay option
    shape = x.get_shape().as_list()
    assert shape[1] == shape[2]
    u_depth = kernel_size[-2]
    u_width = shape[1]
    output_length = kernel_size[3]
    
    with tf.variable_scope(scope_name, reuse=reuse):
        
        if xavier:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        elif variance_scaling:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.variance_scaling_initializer())
        else:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
        
        if spectral_norm:
            weights = u.weights_spectral_norm_conv(weights, update_collection=update_collection, u_width=u_width, 
                                                   u_depth=u_depth, stride=stride, padding=padding, gamma=rho)
        elif l2_norm:
            wd = tf.multiply(tf.nn.l2_loss(weights), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', wd)
        
        conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding=padding)
        
        if use_bias:
            bias = tf.get_variable('bias', output_length, tf.float32, initializer=tf.constant_initializer(0))
            conv = tf.nn.bias_add(conv, bias)
            
        if batch_normal:
            conv = tf.layers.batch_normalization(conv, training=training)
        
        return conv

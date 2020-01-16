#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:54:41 2019
by Arash Rahnama

"""
import tensorflow as tf
import numpy as np

from utility import layers as l

NUM_CLASSES = 10
## this model is adapted from: https://github.com/tensorflow/models/tree/master/official/resnet
def batch_norm(x, training):
    # perform batch normalization 
    # set fused to True for an increase in performance:
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(inputs=x, axis=3, momentum=0.997, 
                                         epsilon=1e-5, center=True, scale=True,
                                         training=training, fused=True)

def fixed_padding(x, kernel_size):
    # perform fixed padding (for input before it's joined in with the output of a block)
    pad_total = kernel_size - 1
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin
    # for (n, h, w, c) format
    padded_x = tf.pad(x, [[0, 0], [pad_begin, pad_end], [pad_begin, pad_end], [0, 0]])
    return padded_x

def conv2d_fixed_padding(x, filters, kernel_size, strides, scope_name='conv2d',
                         l2_norm=False, weight_decay=0,
                         spectral_norm=True, rho=1.0, reuse=None, 
                         update_collection=None):
    # The padding is consistent and is based only on 'kernel_size', not on the
    # dimensions of 'x' (as opposed to using 'tf.layers.conv2d' alone).
    if strides > 1:
        x = fixed_padding(x, kernel_size)
    size_in = x.get_shape().as_list()[-1]
    return l.conv2d(x, [kernel_size, kernel_size, size_in, filters], stride=strides,
                    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                    batch_normal=False, l2_norm=l2_norm, weight_decay=weight_decay, rho=rho,
                    xavier=False, variance_scaling=True, 
                    scope_name=scope_name, spectral_norm=spectral_norm, reuse=reuse, 
                    update_collection=update_collection)
    
def block_sub_function(x, filters, training, projection_feedforward_x, strides, scope_name, 
                       batch_normal=True, l2_norm=False, weight_decay=0,
                       spectral_norm=True, rho=1.0, reuse=None, 
                       update_collection=None):
   
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        feedforward_x = x
        if batch_normal:
            x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)

        if projection_feedforward_x is not None:
            feedforward_x = projection_feedforward_x(x)

        x = conv2d_fixed_padding(x, filters, 3, strides, scope_name='conv1', 
                                 l2_norm=l2_norm, weight_decay=weight_decay,
                                 rho=rho, spectral_norm=spectral_norm, reuse=reuse, 
                                 update_collection=update_collection)
        
        if batch_normal:
            x = batch_norm(x, training)
        x = tf.nn.relu(x)
        
        x = conv2d_fixed_padding(x, filters, 3, 1, scope_name='conv2', 
                                 l2_norm=l2_norm, weight_decay=weight_decay,
                                 rho=rho, spectral_norm=spectral_norm, reuse=reuse, 
                                 update_collection=update_collection)

        return x + feedforward_x

def block_layer_main(x, filters, block_sub_function, blocks, strides, training, scope_name, batch_normal=False, 
                     l2_norm=False, weight_decay=0,
                     spectral_norm=True, rho=1.0, reuse=None, 
                     update_collection=None):
    # creates one layer of the blocks for the ResNet model.
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

        def projection_feedforward_x(x):
            return conv2d_fixed_padding(x, filters, 1, strides,
                                        scope_name='projection_feedforward_x',
                                        l2_norm=l2_norm, weight_decay=weight_decay,
                                        spectral_norm=spectral_norm, rho=rho, 
                                        reuse=reuse, update_collection=update_collection)

        # only the first block per block_layer uses projection_feedforward_x and strides
        x = block_sub_function(x, filters, training, projection_feedforward_x, strides, 'block0',
                               batch_normal=batch_normal, l2_norm=l2_norm, weight_decay=weight_decay,
                               spectral_norm=spectral_norm, rho=rho, reuse=reuse, 
                               update_collection=update_collection)
        
        for i in range(1, blocks):
            x = block_sub_function(x, filters, training, None, 1, 'block{}'.format(i),
                                   batch_normal=batch_normal, l2_norm=l2_norm, weight_decay=weight_decay,
                                   spectral_norm=spectral_norm, rho=rho, reuse=reuse, 
                                   update_collection=update_collection)

        return x

def resnet(x, num_classes=NUM_CLASSES, spectral_norm=True,
           rho_list=[2.52, 2.52, 2.52, 2.52, 2.52],
           batch_normal=False, l2_norm=False, weight_decay=0, 
           update_collection=None, reuse=None, training=False):
    # ResNet Architecture for imagenet.
    # Architecture based on 
    # https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py
    assert len(rho_list) == 6

    num_filters = 64
    kernel_size = 7
    conv_stride = 2
    #resnet_size = 50
    block_sizes = [2, 2, 2, 2] 
    block_strides = [1, 2, 2, 2]
    #first_pool_size=3
    #first_pool_stride=2
    # inputs: a Tensor representing a batch of input images.
    inputs_x = conv2d_fixed_padding(x, num_filters, kernel_size, conv_stride, 
                                    l2_norm=l2_norm, weight_decay=0,
                                    spectral_norm=spectral_norm, rho=rho_list[0], reuse=reuse, 
                                    update_collection=update_collection)

    inputs_x = tf.identity(inputs_x, 'initial_conv')

    inputs_x = tf.identity(inputs_x, 'initial_max_pool')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs_x = block_layer_main(inputs_x, num_filters, block_sub_function, num_blocks, 
                                    block_strides[i], training, scope_name='block_layer{}'.format(i + 1), 
                                    batch_normal=batch_normal, rho=rho_list[i+1],
                                    spectral_norm=spectral_norm, 
                                    reuse=reuse, update_collection=update_collection)

    inputs_x = tf.nn.relu(inputs_x)

    axes = [1, 2]
    inputs_x = tf.reduce_mean(inputs_x, axes, keepdims=True)
    inputs_x = tf.identity(inputs_x, 'final_reduce_mean')
    
    tf.add_to_collection('debug', inputs_x)
    
    output_layer = l.linear(inputs_x, num_classes, scope_name='output_layer', xavier=True,
                            weight_decay=weight_decay, l2_norm=l2_norm, 
                            spectral_norm=spectral_norm, update_collection=update_collection,
                            rho=rho_list[5], reuse=reuse) 


    return output_layer

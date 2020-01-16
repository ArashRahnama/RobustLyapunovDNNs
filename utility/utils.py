#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:54:26 2018
by Arash Rahnama

"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_W = 28
IMAGE_H = 28

def acc(yhat, y):
    # calculate accuracy
    correct_prediction = tf.equal(y, tf.argmax(yhat, 1)) 
    
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def l2_normalize(x, epsilon=1e-12):
    # l2 normalize x   
    x_norm = x/((tf.reduce_sum(x**2)**0.5) + epsilon)
    
    return x_norm     

def loss(yhat, y, mean=True, add_other_losses=True):
    # cross entropy loss
    loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=yhat)
    
    if mean:
        loss_1 = tf.reduce_mean(loss_1)
        
        if add_other_losses:
            tf.add_to_collection('losses', loss_1)
            return tf.add_n(tf.get_collection('losses'))
        
    return loss_1

def latest_epoch(save_dir):
    # return the last epoch number by going through the title of the saved checkpoint files
    return max([int(file_name.split('epoch')[1].split('.')[0])
                for file_name in os.listdir(os.path.join(save_dir, 'checkpoints')) if 'epoch' in file_name])
    
def predict_labels_sess(x, graph, sess, batch_size=100):
    # predict labels in a session
    labels = np.zeros(len(x))
    
    for i in range(0, len(x), batch_size):
        graph_ = sess.run(graph['output_layer'], feed_dict = {graph['input_data']:x[i:i+batch_size]})
        labels[i:i+batch_size] = np.argmax(graph_, 1)
    
    return labels
    
def predict_labels(x, graph, load_dir, batch_size=100, load_epoch=None):
    # run a prediction on the already trained graph and calculate the predictions
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min(latest_epoch(load_dir), load_epoch)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        return predict_labels_sess(x, graph, sess, batch_size=batch_size)
      
def power_iter_lin(u, w, it):
    # perform power iteration method for estimating the largest eigenvalue
    # for a linear feedforward layer
    u_ = u
    
    for i in range(it):
        v_ = l2_normalize(tf.matmul(u_, tf.transpose(w)))
        u_ = l2_normalize(tf.matmul(v_, w))
        
    return u_, v_
    
def power_iter_conv(u, w, it, u_width, u_depth, stride, padding):
    # perform power iteration method for estimating the largest eigenvalue
    # for a convolutional layer  
    u_ = u
    
    for i in range(it):
        v_ = l2_normalize(tf.nn.conv2d(u_, w, strides=[1, stride, stride, 1], padding=padding))
        u_ = l2_normalize(tf.nn.conv2d_transpose(v_, w, [1, u_width, u_width, u_depth], 
                                       strides=[1, stride, stride, 1], padding=padding))
        
    return u_, v_

def cal_weights_spectral_norm_lin_tf(weights, it=30, seed=0):
    # calculates the spectral norm of the weights for a fully connected linear layer
    tf.reset_default_graph()
    
    if seed is not None: 
        tf.set_random_seed(seed)
        
    u = tf.get_variable('u', shape=[1, weights.shape[-1]], 
                        initializer=tf.truncated_normal_initializer(), trainable=False)

    w_ = tf.Variable(weights)
    
    u_hat, v_hat = power_iter_lin(u, w_, it) 
    rho = tf.matmul(tf.matmul(v_hat, w_), tf.transpose(u_hat))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(rho).reshape(-1)
    
def cal_weights_spectral_norm_conv_tf(weights, u_height=IMAGE_H, u_width=IMAGE_W, 
                                      stride=1, it=20, seed=0, padding='SAME'):
    # calculates the spectral norm of the weights in a convolutional layer
    u_depth = weights.shape[-2]
    
    tf.reset_default_graph()
    
    if seed is not None:
        tf.set_random_seed(seed)
    
    u = tf.get_variable('u', shape=[1, u_width, u_width, u_depth], 
                        initializer=tf.truncated_normal_initializer(), trainable=False)

    w = tf.Variable(weights)
    u_hat, v_hat = power_iter_conv(u, w, it=it, u_width=u_width, u_depth=u_depth, 
                                   stride=stride, padding=padding)
    z = tf.nn.conv2d(u_hat, w, strides=[1, stride, stride, 1], padding=padding)
    rho = tf.reduce_sum(tf.multiply(z, v_hat))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(rho).reshape(-1)
        
def weights_spectral_norm_linear(weights, u=None, it=1, update_collection=None, 
                                 reuse=False, gamma=1.0, name='spect_norm_weights_lin'):
    # performs spectral normalization on the weights
    # for a fully connected linear layer
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        weights_shape = weights.get_shape().as_list()

        if u is None:
            u = tf.get_variable('u', shape=[1, weights_shape[-1]], 
                                initializer=tf.truncated_normal_initializer(), trainable=False)

        w_ = tf.reshape(weights, [-1, weights_shape[-1]])
        u_hat, v_hat = power_iter_lin(u, w_, it)
        rho = tf.maximum(tf.matmul(tf.matmul(v_hat, w_), tf.transpose(u_hat))/gamma, 1)
            
        w_ = w_/rho

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_, weights_shape)
        else:
            tf.add_to_collection(update_collection, u.assign(u_hat))
            w_norm = tf.reshape(w_, weights_shape)

        tf.add_to_collection('weights_after_spectral_norm_lin', w_norm)

        return w_norm
    
def weights_spectral_norm_conv(weights, u=None, it=1, update_collection=None, reuse=False,
                               gamma=1.0, name='spect_norm_weights_conv', u_width=IMAGE_W, 
                               u_depth=3, stride=1, padding='SAME'):
    # performs spectral normalization on the weights for a convolutional layer
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        
    if u is None:
        u = tf.get_variable('u', shape=[1, u_width, u_width, u_depth], 
                            initializer=tf.truncated_normal_initializer(), trainable=False)
    
    u_hat, v_hat = power_iter_conv(u, weights, it=it, u_width=u_width, u_depth=u_depth, stride=stride, padding=padding)
    z = tf.nn.conv2d(u_hat, weights, strides=[1, stride, stride, 1], padding=padding)
    rho = tf.maximum(tf.reduce_sum(tf.multiply(z, v_hat))/gamma, 1)
    
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = weights/rho
    else:
        tf.add_to_collection(update_collection, u.assign(u_hat))
        w_norm = weights/rho
    
    tf.add_to_collection('weights_after_spectral_norm_conv', w_norm)

    return w_norm       
    
def save_image(img, save_dir):
    # save images into the save_dir directory
    for i in range(len(img)):
        fig = np.around((img[i] + 0.5)*255)
        fig = fig.astype(np.uint8).squeeze()
        pic = Image.fromarray(fig)
        pic.save(os.path.join(save_dir,"img" + str(i) + ".png"))
    return

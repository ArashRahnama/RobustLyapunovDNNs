#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:50:58 2018
by Arash Rahnama

"""
import numpy as np
import tensorflow as tf
from utility import utils as u

NUM_CLASSES = 10
NUM_CHANNELS = 3
IMAGE_H = 28#32#224

def tf_l1_norm(x, epsilon=1e-24):
    # calculate the L1 norm
    reduction_indices = list(range(1, len(x.get_shape())))
    
    return tf.reduce_sum(tf.abs(x), reduction_indices=reduction_indices, keep_dims=True) + epsilon

def tf_l2_norm(x, epsilon=1e-24):
    # calculate the L2 norm
    reduction_indices = list(range(1, len(x.get_shape())))
    
    return tf.sqrt(tf.reduce_sum(tf.square(x), reduction_indices=reduction_indices, keep_dims=True)) + epsilon

def project_back_onto_unit_ball(x_adv, x, eps=0.3, norm=2):
    # project x_adv to an eps-ball around x
    # min ||x_adv-x|| subject to x_adv in epsilon
    delta = x_adv-x
    
    if norm == 1:
        norms = tf_l1_norm(delta)
    if norm == 2:
        norms = tf_l2_norm(delta)
        
    proj_denominator = tf.maximum(tf.ones_like(norms), norms/eps)
    
    return x+delta/proj_denominator

def fgm(x, preds, labs=None, eps=0.3, norm=2, clip_min=None, clip_max=None, **kwargs): 
    # tensorflow version of the Fast Gradient Method inspired by,
    # https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
    # x -> the input placeholder
    # preds -> the model's output tensor
    # labs (optional) -> a placeholder for the model labels. Provide this parameter, 
    # if the goal is to use true labels to craft adversarial samples. Otherwise,
    # model predictions are used as labels to avoid the "label leaking" effect 
    # (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
    # eps -> the epsilon (input variation parameter)
    # norm (optional) -> norm of the attack (mimics Numpy, possible values: np.inf, 1 or 2).
    # clip_min -> Minimum float value for adversarial example components
    # clip_max -> Maximum float value for adversarial example components
    # return -> a tensor for the adversarial example
    if labs is None:
        # using model predictions as ground truth to avoid label leaking
        labs = tf.argmax(preds, 1)

    # compute loss (without taking the mean across samples)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labs, logits=preds)

    # define gradient of loss wrt input
    grad, = tf.gradients(loss_, x)
    
    if norm == np.inf:
        # take sign of gradient
        signed_grad = tf.sign(grad)   
    elif norm == 1:
        signed_grad = grad / tf_l1_norm(grad)
    elif norm == 2:
        signed_grad = grad / tf_l2_norm(grad)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are currently implemented.")
 
    # multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad
    
    # add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)
    
    # if clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        
    return adv_x

def pgdm(x, preds, y=None, eps=0.3, norm=2, model=None, a=None, k=100, weight_decay=0, 
         l2_norm=False, spectral_norm=True, reuse=True, update_collection='_', 
         rho_list=[1.0,1.0,1.0], num_classes=NUM_CLASSES, batch_normal=False, 
         training=False):
    # tensorflow version of the Projected Gradient Descent Method inspired by,
    # https://github.com/duchi-lab/certifiable-distributional-robustness/blob/master/attacks_tf.py
    # x -> the input placeholder
    # preds -> the model's output tensor
    # y (optional) -> a placeholder for the model labels. Provide this parameter, 
    # if the goal is to use true labels to craft adversarial samples. Otherwise,
    # model predictions are used as labels to avoid the "label leaking" effect 
    # (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
    # eps -> the epsilon (input variation parameter)
    # k -> number of steps to take, each of size a
    # a -> size of each step
    # model -> tensorflow graph model (**kwargs goes to this)
    # norm (optional) -> norm of the attack (mimics Numpy, possible values: 1 or 2).
    # return -> a tensor for the adversarial example
    if a is None:
        a = 2.*eps/k

    if y is None:
        # using model predictions as ground truth to avoid label leaking
        y = tf.argmax(preds, 1)
    
    x_adv = x
    
    for t in range(k):
        loss_ = u.loss(model(x_adv, reuse=reuse, rho_list=rho_list, update_collection=update_collection, 
                             weight_decay=weight_decay, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                             batch_normal=batch_normal, num_classes=num_classes), y, mean=False)
        
        grad, = tf.gradients(loss_, x_adv)
        
        if norm == 1:
            scaled_grad = grad / tf_l1_norm(grad)  
        elif norm == 2:
            scaled_grad = grad / tf_l2_norm(grad)   
        elif norm == np.inf:
            scaled_grad = tf.sign(grad)
        
        x_adv = tf.stop_gradient(x_adv + a*scaled_grad)
        
        if norm in [1, 2]:
            x_adv = project_back_onto_unit_ball(x_adv, x, eps=eps, norm=norm)
        elif norm == np.inf:
            x_adv = tf.clip_by_value(x_adv, x-eps, x+eps)
        
    return x_adv

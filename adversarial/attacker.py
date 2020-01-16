#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:08:53 2018
by Arash Rahnama

"""
import time
import os
import numpy as np
from numpy import random
import tensorflow as tf

from utility import graph as g
from utility import utils as u
from adversarial import attacks

NUM_CHANNELS = 3
NUM_CLASSES = 10

def generate_adversary_sess(x, y, graph, sess, batch_size=100, method=attacks.fgm, num_classes=NUM_CLASSES, 
                            GPU_id=0, l2_norm=False, spectral_norm=True, weight_decay=0, batch_normal=False, **kwargs):
    # use the trained model and attack model to generate adversarial examples from the dataset x  
    if method == 'cw':
        attack = attacks.CarliniL2(sess, graph, batch_size=9, max_iterations=1000, confidence=0)
        adv_x = attack.attack(x, y)
    else:
        adv_tensor = method(graph['input_data'], graph['output_layer'], num_classes=num_classes, batch_normal=batch_normal,
                            l2_norm=l2_norm, spectral_norm=spectral_norm, weight_decay=weight_decay, **kwargs)
    
    adv_x = np.zeros(np.shape(x))

    for i in range(0, len(x), batch_size):
        adv_x[i:i+batch_size] = sess.run(adv_tensor, feed_dict={graph['input_data']: x[i:i+batch_size]})
        
    return adv_x

def attack_main(x, arch, load_dir, num_classes=NUM_CLASSES, rho_list=[1.0,1.0,1.0], num_channels=NUM_CHANNELS, 
                GPU_id=0, load_epoch=None, method=attacks.fgm, l2_norm=False, batch_normal=False, 
                spectral_norm=True, weight_decay=0, defense='none', **kwargs):
    # the main function to build the deep learning graph nad generate adversarial examples
    if load_epoch is None:
        load_epoch = u.latest_epoch(load_dir)
    else:
        load_epoch = np.min(u.latest_epoch(load_dir), load_epoch)
        
    tf.reset_default_graph()
    with tf.device("/GPU:%s"%(GPU_id)):
        graph = g.graph_wrapper(arch, num_classes=num_classes, rho_list=rho_list, weight_decay=weight_decay, 
                                l2_norm=l2_norm, spectral_norm=spectral_norm, num_channels=num_channels, adv=defense,
                                update_collection='_', batch_normal=batch_normal, save_dir=load_dir)
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints','epoch%s'%(load_epoch)))
        
            return generate_adversary_sess(x, graph, sess, method=method, model=arch, rho_list=rho_list, 
                                           weight_decay=weight_decay, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                                           batch_normal=batch_normal, num_classes=num_classes, **kwargs)
        
def test_model_robustness(x, y, arch, load_dir, save_adv_img, rho_list=[1.0,1.0,1.0], num_channels=NUM_CHANNELS, 
                          verbose=True, load_epoch=None, remove_zero_adv=True, num_classes=NUM_CLASSES, defense='none',
                          method=attacks.fgm, opt_method='momentum', GPU_id=0, l2_norm=False, spectral_norm=True, 
                          weight_decay=0, batch_normal=False, **kwargs):
    # test the robustness of a trained network by generating adversarially perturbed examples and calculating the accuracy
    tf.reset_default_graph()

    start = time.time()
    
    with tf.device("/GPU:%s"%(GPU_id)):
        
        graph = g.graph_wrapper(arch, num_classes=num_classes, save_dir=load_dir, rho_list=rho_list, 
                                num_channels=num_channels, weight_decay=weight_decay, l2_norm=l2_norm, 
                                adv=defense, batch_normal=batch_normal, spectral_norm=spectral_norm, 
                                update_collection='_', opt_method=opt_method)
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:      
            
            if load_epoch is None:
                load_epoch = u.latest_epoch(load_dir)
            else:
                load_epoch = np.min(u.latest_epoch(load_dir), load_epoch)

            graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        
            # generate adversarial examples
            adv_x = generate_adversary_sess(x, y, graph, sess, method=method, model=arch, rho_list=rho_list,
                                            weight_decay=weight_decay, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                                            batch_normal=batch_normal, num_classes=num_classes, **kwargs)
        
            # remove adversarial examples that are equal to zero
            if remove_zero_adv:
                reduction_ind = tuple(range(1, len(x.shape)))
                mag_delta = np.sqrt(np.sum(np.square(adv_x-x), axis=reduction_ind))
                keep_inds = mag_delta > 1e-4
                if np.sum(keep_inds) > 0:
                    adv_x, y = adv_x[keep_inds], y[keep_inds]
            # save samples of adversarial examples first time the code is run
#            if(not os.path.exists(os.path.join(save_adv_img,"img" + str(0) + ".png"))): 
#                indices = random.choice(len(adv_x), 50)
#                u.save_image(adv_x[indices], save_adv_img)

            yhat_adv = u.predict_labels_sess(adv_x, graph, sess)

    acc_adv = np.sum(yhat_adv == y)/float(len(y))

    if verbose:
        print('Accuracy on adversarial samples: %.4f (%.3f seconds elapsed)'%(acc_adv, time.time()-start))
        
    return acc_adv

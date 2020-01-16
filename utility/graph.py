#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:20:30 2018
by Arash Rahnama

"""
import time
import os
import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle

from adversarial import attacks as att
from utility import utils as u

IMAGE_H = 28
IMAGE_W = 28
NUM_CHANNELS = 3
NUM_CLASSES = 10

def graph_wrapper(arch, num_classes=NUM_CLASSES, adv='basic', eps=0.3, weight_decay=0, update_collection=None, 
                  rho_list=[1.0,1.0,1.0], save_histograms=True, num_channels=NUM_CHANNELS, max_save=200, batch_normal=False,
                  l2_norm=False, spectral_norm=True,training=False, norm=2, opt_method='momentum', save_dir=None):
    # build the graph (wrapper for all variables)
    input_data = tf.placeholder(tf.float32, shape=[None, IMAGE_H, IMAGE_W, num_channels], name='InputData')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='InputLabels')
    
    output_layer = arch(input_data, num_classes=num_classes, weight_decay=weight_decay, batch_normal=batch_normal,
                        training=training, l2_norm=l2_norm, spectral_norm=spectral_norm, rho_list=rho_list, 
                        update_collection=update_collection)
    
    # define the optimization approach and the loss 
    learning_rate = tf.Variable(0.01, name='learning_rate', trainable=False)
    
    if adv == 'fgm':
        x_adv = att.fgm(input_data, output_layer, eps=eps, norm=norm, training=training)
        output_layer_final = arch(x_adv, num_classes=num_classes, weight_decay=weight_decay, rho_list=rho_list, 
                                  l2_norm=l2_norm, spectral_norm=spectral_norm, update_collection=update_collection, 
                                  batch_normal=batch_normal, reuse=True, training=training)
    elif adv == 'pgdm':
        x_adv = att.pgdm(input_data, output_layer, eps=eps, norm=norm, model=arch, k=15, weight_decay=weight_decay,
                        l2_norm=l2_norm, spectral_norm=spectral_norm, num_classes=num_classes, rho_list=rho_list, 
                        training=training)
        output_layer_final = arch(x_adv, num_classes=num_classes, weight_decay=weight_decay, rho_list=rho_list, 
                                  l2_norm=l2_norm, spectral_norm=spectral_norm, update_collection=update_collection, 
                                  batch_normal=batch_normal, reuse=True, training=training)
    else:
        output_layer_final = output_layer
    
    total_loss =u.loss(output_layer_final, input_labels)
    total_acc = u.acc(output_layer_final, input_labels)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if opt_method == 'adam' or num_channels == 1:
            opt_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)
        else:
            opt_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    ###################################################################
    ## graph's output dictionary 
    graph = dict(
            input_data = input_data, 
            input_labels = input_labels, 
            total_loss = total_loss, 
            total_acc = total_acc, 
            output_layer = output_layer, 
            output_layer_final = output_layer_final, 
            opt_step = opt_step, 
            learning_rate = learning_rate
            )
    ###################################################################
    ## saving the graph information to tensorboard
    if save_dir is not None:
        saver = tf.train.Saver(max_to_keep=max_save)
        graph['saver'] = saver
        
        if not os.path.isdir(save_dir):
            tf.summary.scalar('loss', total_loss)
            tf.summary.scalar('accuracy', total_acc)
            
            ## histograms for variables that are being trained
            if save_histograms:
                for i in tf.trainable_variables():
                    tf.summary.histogram(i.op.name, i)
            
            ## merge the summaries and write them to save_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'))
            graph_writer = tf.summary.FileWriter(os.path.join(save_dir, 'graph'), 
                                                 graph=tf.get_default_graph())
            valid_writer = tf.summary.FileWriter(os.path.join(save_dir, 'validation'))
            
            graph['merged'] = merged
            graph['train_writer'] = train_writer
            graph['graph_writer'] = graph_writer
            graph['valid_writer'] = valid_writer
     ###################################################################           
    return graph

def graph_train(x, y, graph, save_dir, val_data=None, val_labels=None, initial_learning_rate=0.01, 
                num_epochs=100, batch_size=64, write_freq=10, verbose=True, load_epoch=-1, 
                batch_normal=False, early_stop_acc=None, early_stop_acc_num=10, seed=0):
    ## function to train the graph
    np.random.seed(seed)
    tf.set_random_seed(seed)

    save_step = 10
        
    start_time = time.time()
    
    training_losses, training_accs = [], []
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        if load_epoch>-1:
            if verbose:
                print('Continuing training starting at epoch %s+1'%(load_epoch))
            if save_dir is not None:
                restore_weights_file = os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch))
            if 'saver' in graph:
                graph['saver'].restore(sess, restore_weights_file)
        else:
            if save_dir is not None and not os.path.exists(os.path.join(save_dir, 'checkpoints')):
                os.mkdir(os.path.join(save_dir, 'checkpoints'))
            if 'saver' in graph and save_dir is not None:
                graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch0'))
        
        for epoch in range(load_epoch+2, load_epoch+num_epochs+2):
            
            learning_rate = initial_learning_rate*(0.95**(epoch/390)) ## rate of learning rate change
            sess.run(graph['learning_rate'].assign(learning_rate))
            
            start_time2 = time.time()
            training_loss = 0
            training_acc = 0
            steps = 0
            
            x_, y_ = shuffle(x, y)
            
            if len(x_)%batch_size == 0:
                end_ind = len(x_)
            else:
                end_ind = len(x_)-batch_size
            # training loop    
            for i in range(0, end_ind, batch_size):
                x_train, y_train = x_[i:i+batch_size], y_[i:i+batch_size]
                feed_dict = {graph['input_data']: x_train, graph['input_labels']: y_train}
                training_loss_, training_acc_, _ = sess.run([graph['total_loss'], graph['total_acc'], graph['opt_step']], 
                                                            feed_dict=feed_dict)
                training_loss += training_loss_
                training_acc += training_acc_
                steps += 1
                
                if verbose:
                    print('\rEpoch %s/%s (%.3f seconds), batch %s/%s (%.3f seconds): loss %.3f, acc %.3f'
                          %(epoch, load_epoch+num_epochs+1, time.time()-start_time, steps, 
                            len(x_)/batch_size, time.time()-start_time2, training_loss_, training_acc_),
                            end='')
            # write to tensorboard                 
            if 'saver' in graph and epoch%write_freq == 0:
                summary = sess.run(graph['merged'], feed_dict=feed_dict)
                graph['train_writer'].add_summary(summary, epoch)
            
                if val_data is not None:
                    feed_dict = {graph['input_data']: val_data, 
                                 graph['input_labels']: val_labels}
                    summary = sess.run(graph['merged'], feed_dict=feed_dict)
                    graph['valid_writer'].add_summary(summary, epoch)
        
            if 'saver' in graph and save_dir is not None and epoch%save_step == 0:
                graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))
            
            training_losses.append(training_loss/float(steps))
            training_accs.append(training_acc/float(steps))
        
            if early_stop_acc is not None and np.mean(training_accs[-early_stop_acc_num:]) >= early_stop_acc:
                if verbose:
                    print('\rMean accuracy >= %s for the last %s epochs. Stopping training after the epoch %s/%s.'
                          %(early_stop_acc, early_stop_acc_num, epoch, load_epoch+num_epochs+1), end='')
                break
        
        if verbose:
            print('\nFINISHED: The model was trained for %s epochs.'%(epoch))
        
        if 'saver' in graph and save_dir is not None and not os.path.exists(os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch))):
                    graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))
                    
    return training_losses, training_accs
                    
def graph_build(x, y, arch, save_dir, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, adv='none', eps=0.3, 
                weight_decay=0, batch_normal=False, l2_norm=False, spectral_norm=True, verbose=True, 
                rho_list=[1.0,1.0,1.0], norm=2, opt_method='momentum', GPU_id=0, **kwargs):
    # build the tensorflow graph and train
    tf.reset_default_graph()
    
    if verbose:
        start_time = time.time()
    
    with tf.device("/GPU:%s"%(GPU_id)):   
        if save_dir is None or not os.path.exists(save_dir) or 'checkpoints' not in os.listdir(save_dir):
            graph = graph_wrapper(arch, adv=adv, eps=eps, num_classes=num_classes, weight_decay=weight_decay, 
                                  batch_normal=batch_normal, num_channels=num_channels, rho_list=rho_list, 
                                  norm=norm, training=True, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                                  opt_method=opt_method, save_dir=save_dir)
            train_losses, train_accrucies = graph_train(x, y, graph, save_dir, **kwargs)
        else:
            graph = graph_wrapper(arch, num_classes=num_classes, weight_decay=weight_decay, batch_normal=batch_normal,
                                  l2_norm=l2_norm, spectral_norm=spectral_norm, num_channels=num_channels, 
                                  rho_list=rho_list, norm=norm, update_collection='_', opt_method=opt_method, 
                                  save_dir=save_dir)
            if verbose:
                print('Model already exists and is being loaded!\n')
        
        if save_dir is None:
            train_accuracy = np.nan
            if verbose:
                print('The given save_dir is None.. returning NaN since the weights are not saved.\n')
        else:
            yhat = u.predict_labels(x, graph, save_dir)
            train_accuracy = np.sum(yhat == y)/float(len(y))
    
    if verbose:
        print('Train accuracy: %.2f (%.1f seconds elapsed)\n'%(train_accuracy, time.time()-start_time))
    
    return train_accuracy

def graph_predict(x, arch, load_dir, labs=None, num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, delta=1.0, 
                  batch_normal=False, l2_norm=False, spectral_norm=True, weight_decay=0, load_epoch=None, norm=2, 
                  opt_method='momentum', GPU_id=0):
    # build the tensorflow graph and predict labels
    tf.reset_default_graph()
    with tf.device("/GPU:%s"%(GPU_id)):   
        graph = graph_wrapper(arch, num_classes=num_classes, num_channels=num_channels, delta=1.0, 
                              batch_normal=batch_normal, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                              weight_decay=weight_decay, norm=norm, update_collection='_',
                              opt_method='momentum', save_dir=load_dir)
    
        labs_hat = u.predict_labels(x, graph, load_dir, load_epoch=load_epoch)
    
    if labs is None:
        return labs_hat
    
    return np.sum(labs_hat == labs)/float(len(labs))

def graph_accuracy(x, y, arch, adv='none', eps=0.3, rho_list=[1.0,1.0,1.0], norm=2, l2_norm=False, 
                   spectral_norm=True, weight_decay=0, batch_size=100, batch_normal=False, load_epoch=None, 
                   num_channels=NUM_CHANNELS, opt_method='momentum', save_dir=None):
    # build the tensorflow graph and produce its accuracy
    tf.reset_default_graph()
    
    graph = graph_wrapper(arch, adv=adv, eps=eps, num_channels=num_channels, rho_list=rho_list, 
                          weight_decay=weight_decay, l2_norm=l2_norm, spectral_norm=spectral_norm, 
                          batch_normal=batch_normal, norm=norm, update_collection='_', 
                          opt_method=opt_method, save_dir=save_dir)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        load_file = tf.train.latest_checkpoint(os.path.join(save_dir, 'checkpoints'))
        if load_epoch is not None:
            load_file = load_file.replace(load_file.split('epoch')[1], str(load_epoch))
        graph['saver'].restore(sess, load_file)
        
        correct = 0
        total = 0
        
        for i in range(0, len(x), batch_size):
            x_test, y_test = x[i:i+batch_size], y[i:i+batch_size]
            num_batch_samples = len(x_test)
            feed_dict = {graph['input_data']: x_test, graph['input_labels']: y_test}
            correct += sess.run(graph['total_acc'], feed_dict=feed_dict)*num_batch_samples
            total += num_batch_samples
            
        return correct/total

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:52:39 2018
by Arash Rahnama

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse

from data.load_mnist import MNIST
from data.load_cifar import CIFAR
from data.load_svhn import SVHN

from utility import graph as g
from models import nnets as n
from adversarial import attacks as a1
from adversarial import attacker as a2
from models import resnet as r
# root directory for the project
ROOT_DIR = os.getcwd()
# directory in which we save the trained models including their weights and biases
MODEL_DIR = os.path.join(ROOT_DIR, "model_data_cifar/") 
# directory in which we save the sample images from the original test dataset
SAVE_ORIGINAL_IMG = None
# directory in which we save the sample images from the adversarial test dataset
SAVE_ADV_IMG = None

NUM_CLASSES = 10
NUM_CHANNELS = 3
###############################################################################
def initialize_training(x, y, arch, save_dir, val_data=None, val_labels=None, l2_norm=False, spectral_norm=True,
                        rho_list=[1.0, 1.0, 1,0], adv=None, norm=2, eps=0.3,  opt_method='momentum', batch_normal=False, 
                        initial_learning_rate=0.01, weight_decay=0, num_epochs=200, write_freq=25):
    # initialize the training
    # x -> the training dataset
    # y -> the training labels
    # val_data -> validation dataset
    # val_labels -> validation labels
    # arch -> the deep learning model (forwardnet, alexnet etc.)
    # save_dir -> the directory to save the weights, biases and tensorboard metadata for a specific model
    # adv -> string indicating the adversarial training scheme ('basic', 'fgsm', 'pgdm' or 'wrm')
    # basic indicates no adversarial training
    # norm -> the norm of the attack (np.inf, 1 or 2 for FGM or PGD)
    # eps -> the magnitude of the attack during training (the epsilon)
    # opt_method -> the method of optimization used in training ('adam', 'momentum')
    # initial_learning_rate -> initial learning rate used in the training
    # num_epochs -> number of epochs
    # write_freq -> save the metadata every this many epochs
    # spectral_norm -> enables spectral normalization of layers
    # l2_norm -> enables l2-norm regularization of layers
    # weight_decay -> l2-norm regularization parameter
    # batch_normal -> enables batch normalization for convolutional layers
    num_channels = x.shape[-1]

    print('The magnitude of the attack is eps = %.4f, saving the metadata to %s \n'%(eps, save_dir))
    print('Training starts..\n')
    
    _ = g.graph_build(x, y, arch, save_dir, val_data=val_data, val_labels=val_labels, 
                      num_channels=num_channels, rho_list=rho_list, adv=adv, norm=norm, batch_normal=batch_normal,
                      eps=eps, opt_method=opt_method, initial_learning_rate=initial_learning_rate, 
                      num_epochs=num_epochs, weight_decay=weight_decay, write_freq=write_freq, batch_size=128, 
                      l2_norm=l2_norm, spectral_norm=spectral_norm, early_stop_acc=0.999, early_stop_acc_num=5)
###############################################################################
def initialize_attack(x, y, arch, eps, attack, load_dir, save_adv_img, defense='basic', l2_norm=False, spectral_norm=True, 
                      batch_normal=False, rho_list=[1.0, 1.0, 1,0], delta=0.85, weight_decay=0, load_epoch=None, norm=2, 
                      opt_method='momentum'): 
    # initialize the attack
    # x -> the test data
    # y -> the test labels
    # arch -> the trained deep learning model
    # eps -> the magnitude of the attack during testing
    # defense -> if adversarial training was used during training, the type of adversarial training i.e. 'basic', 'fgsm', 'pgdm' or 'wrm'
    # attack -> the type of attack i.e. 'fgsm', 'pgdm' or 'wrm'
    # load_dir -> the directory in which the metadata for the trained model is saved in
    # save_adv_img -> the directory in which the adversarial image samples will be saved in
    # save_result_file -> pickled file containing the results will be saved in this directory
    # gamma -> the gamma to test
    # load_epoch -> the epoch corresponding to the model to load
    # norm -> the norm of the attack (np.inf, 1 or 2 for FGSM or PGDM)
    # opt_method -> the method of optimization used ('adam', 'momentum')
    # returns a the accuracy of the tested model with the given beta against the given attack with the given epsilon 
    # indicating the performance at the epsilon level of the specific attack
    # spectral_norm -> enables spectral normalization of layers
    # l2_norm -> enables l2-norm regularization of layers
    # weight_decay -> l2-norm regularization parameter
    # batch_normal -> enables batch normalization for convolutional layers
    num_channels = x.shape[-1]

    trained_model_dir = os.path.join(load_dir, 'defense_%s_Lya_delta%s'%(defense, delta))
    adv_accs = a2.test_model_robustness(x, y, arch, trained_model_dir, save_adv_img, l2_norm=l2_norm, 
                                        rho_list=rho_list, method=attack, load_epoch=load_epoch, 
                                        batch_normal=batch_normal, num_channels=num_channels, norm=norm, 
                                        spectral_norm=spectral_norm, opt_method=opt_method, eps=eps, 
                                        defense=defense, weight_decay=weight_decay)
        
    return adv_accs
###############################################################################       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments in 'Robust Design of Neural Networks against Adversarial Attacks based on Lyapunov Theory' written by Arash Rahnama")
    parser.add_argument('--dataset', 
                required=False,
                default="cifar",
                choices=["mnist", "cifar", "svhn"],
                help='The dataset to be used for the experiment.')
    
    args = parser.parse_args()

    if args.dataset == "mnist":
        data = MNIST(os.path.join(ROOT_DIR,"data/data_mnist"))
    elif args.dataset == "cifar":
        data = CIFAR(os.path.join(ROOT_DIR,"data/data_cifar"))
        data.train_data = data.train_data[:, 2:30, 2:30, :]
        data.validation_data = data.validation_data[:, 2:30, 2:30, :]
        data.test_data = data.test_data[:, 2:30, 2:30, :]
    elif args.dataset == "svhn":
        data = SVHN(os.path.join(ROOT_DIR,"data/data_svhn"))
    else:
        raise(RuntimeError("unknown dataset: "+args.datset))
     
    arch = n.alexnet #n.forward_net r.resnet
    
    rho_list = [2.52, 2.52, 2.52, 2.52, 2.02] 
    #delta's 0.74, 0.74, 0.74, 0.74, 0.86
    #nu's 0.33, 0.33, 0.33, 0.33, 0.29 
    delta = 0.85
    
    epsilon = 0.2 # the attack's strength
    defense = 'none'  # for adversarial training you can set this to fgm or pgdm
    
    if not os.path.exists(MODEL_DIR): 
        os.mkdir(MODEL_DIR)
    save_dir = os.path.join(MODEL_DIR, 'defense_%s_Lya_delta%s'%(defense, delta)) # where the models meta data is saved in
    initialize_training(data.train_data, data.train_labels, arch, save_dir, data.validation_data[:100], 
                        data.validation_labels[:100], eps=0, rho_list=rho_list, spectral_norm=True,
                        l2_norm=False, adv='none', weight_decay=0)

    ## initialize the attack and report the accuracy on the adversarial test data set
    adv_results = initialize_attack(data.test_data, data.test_labels, arch, epsilon, a1.fgm,
                                    MODEL_DIR, SAVE_ADV_IMG, defense=defense, norm=2, 
                                    delta=delta, rho_list=rho_list)
    print('The magnitude of the attack is eps = %.4f, the robustness index of the model being tested is delta = %.4f'%(epsilon, delta))
    print('The model\'s accuracy on the adversarial dataset is', adv_results,'\n')

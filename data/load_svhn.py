## setup_cifar.py -- cifar data and model loading code
##
## The code for loading the SVHN dataset 
##
from scipy.io import loadmat
import numpy as np
import os

TRAIN_DATASET_SIZE = 73257
TEST_DATASET_SIZE = 26032
VALIDATION_SIZE = 500

def load_data(file_path):
    data = loadmat(file_path)
    return ((data['X']/255)-0.5), data['y']

class SVHN:
    def __init__(self, data_path):
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        
        if not os.path.exists(data_path):
            print("cannot access the folder containing the data")

        File_Name = "train_32x32.mat"
        r,s = load_data(os.path.join(data_path, File_Name))
        train_data.extend(r)
        train_labels.extend(s)
            
        train_data = np.array(train_data, dtype=np.float32)#,dtype=np.uint8)
        train_labels = np.array(train_labels)
        ### dimension*dimension, channel, length-> length, dimension*dimension, channel
        train_data = train_data.transpose((3,0,1,2)) 
        train_labels = train_labels[:,0]
        ### change the labels range from 1-10 to 1-9
        train_labels[train_labels == 10] = 0

        File_Name = "test_32x32.mat"
        r_t,s_t = load_data(os.path.join(data_path, File_Name))
        test_data.extend(r_t)
        test_labels.extend(s_t)
            
        test_data = np.array(test_data, dtype=np.float32)#,dtype=np.uint8)
        test_labels = np.array(test_labels)
        ### dimension*dimension, channel, length-> length, dimension*dimension, channel
        test_data = test_data.transpose((3,0,1,2)) 
        test_labels = test_labels[:,0]
        ### change the labels range from 1-10 to 1-9
        test_labels[test_labels == 10] = 0

        self.test_data = test_data
        self.test_labels = test_labels
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

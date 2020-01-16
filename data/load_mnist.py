## The code for loading the MNIST dataset 
##
import numpy as np
import os
import gzip

TRAIN_DATASET_SIZE = 60000
TEST_DATASET_SIZE = 10000
VALIDATION_SIZE = 5000
IMAGE_SIZE = 28
CHANNEL_NUM = 1

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*IMAGE_SIZE*IMAGE_SIZE)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    return np.argmax(labels, axis=1)

class MNIST:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            print("cannot access the folder containing the data")
            
        train_data = extract_data(os.path.join(data_path, "train-images-idx3-ubyte.gz"), TRAIN_DATASET_SIZE)
        train_labels = extract_labels(os.path.join(data_path, "train-labels-idx1-ubyte.gz"), TRAIN_DATASET_SIZE)
        
        self.test_data = extract_data(os.path.join(data_path, "t10k-images-idx3-ubyte.gz"), TEST_DATASET_SIZE)
        self.test_labels = extract_labels(os.path.join(data_path,"t10k-labels-idx1-ubyte.gz"), TEST_DATASET_SIZE)
        
    
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

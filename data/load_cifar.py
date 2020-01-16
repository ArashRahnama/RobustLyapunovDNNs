## The code for loading the CIFAR dataset 
##
import numpy as np
import os

VALIDATION_SIZE = 5000
TEST_DATASET_SIZE = 10000
NUM_FILES = 5
IMAGE_SIZE = 32
CHANNEL_NUM = 3

def load_batch(file_path):
    f = open(file_path,"rb").read()
    size = IMAGE_SIZE*IMAGE_SIZE*CHANNEL_NUM+1
    labels = []
    images = []
    for i in range(TEST_DATASET_SIZE):
        arr = np.fromstring(f[i*size:(i+1)*size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((CHANNEL_NUM, IMAGE_SIZE, IMAGE_SIZE)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
        
    return np.array(images), np.argmax(np.array(labels), axis=1)
  

class CIFAR:
    def __init__(self, data_path):
        train_data = []
        train_labels = []
        
        if not os.path.exists(data_path):
            print("cannot access the folder containing the data")

        for i in range(NUM_FILES):
            File_Name = "data_batch_"+str(i+1)+".bin"
            r,s = load_batch(os.path.join(data_path, File_Name))
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = load_batch(os.path.join(data_path, "test_batch.bin"))
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

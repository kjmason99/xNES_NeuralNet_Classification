# Code contains functions to read in data and perform data manipulation such as
# normalizing the data, dividing into training /testing etc.

import numpy as np
from csv import reader
import random


# Read csv file as X, y data (last column is class label)

def read_csv(filename):
    X_str = list()  # features
    y_str = list()  # labels

    # open file
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            # Check if row is empty
            if not row:
                continue
            else:
                X_str.append(row[:-1])
                y_str.append(row[-1])
                

    # Convert class labels into distinct classes
    def convert_str2idx(y_str):
        unique = set(y_str)
        lookup = dict()
        
        for idx_label, label in enumerate(unique):
            lookup[label] = idx_label
        y_idx = list()
        for label in y_str:
            y_idx.append(lookup[label])
        return y_idx

    y_idx = convert_str2idx(y_str)

    # Convert to numpy arrays
    X = np.array(X_str, dtype=np.float32)
    y = np.array(y_idx, dtype=np.int)


    return (X, y)

# split data into training and testing data sets
def split_data(X_data, y_data, weight):

        X_train_size = int(X_data.shape[0] * weight)
        X_test_size = int(X_data.shape[0] - X_train_size)
        y_train_size = X_train_size
        y_test_size = X_test_size
        
        X_train_set = np.zeros(shape=(X_train_size,X_data.shape[1]))
        X_test_set = np.zeros(shape=(X_test_size,X_data.shape[1]))
        y_train_set = np.zeros(shape=(y_train_size),dtype=int)
        y_test_set = np.zeros(shape=(y_test_size),dtype=int)

        for i in range(X_train_size):
            index = random.randrange(X_data.shape[0])
            X_train_set[i] = X_data[index]
            y_train_set[i] = y_data[index]
            X_data = np.delete(X_data, index, axis=0)
            y_data = np.delete(y_data, index, axis=0)
            

        return [X_train_set, X_data,y_train_set,y_data]

# convert data into training, testing and validation data sets
def split_data_validation(X_data, y_data, train_frac, val_frac):
        X_train_size = int(X_data.shape[0] * train_frac)
        X_val_size = int(X_data.shape[0] * val_frac)
        X_test_size = int(X_data.shape[0] - (X_train_size + X_val_size))
        y_train_size = X_train_size
        y_val_size = X_val_size
        y_test_size = X_test_size
        
        X_train_set = np.zeros(shape=(X_train_size,X_data.shape[1]))
        X_val_set = np.zeros(shape=(X_val_size,X_data.shape[1]))
        X_test_set = np.zeros(shape=(X_test_size,X_data.shape[1]))
        y_train_set = np.zeros(shape=(y_train_size),dtype=int)
        y_val_set = np.zeros(shape=(y_val_size),dtype=int)
        y_test_set = np.zeros(shape=(y_test_size),dtype=int)

        for i in range(X_train_size):
            index = random.randrange(X_data.shape[0])
            X_train_set[i] = X_data[index]
            y_train_set[i] = y_data[index]
            X_data = np.delete(X_data, index, axis=0)
            y_data = np.delete(y_data, index, axis=0)
            
        for i in range(X_val_size):
            index = random.randrange(X_data.shape[0])
            X_val_set[i] = X_data[index]
            y_val_set[i] = y_data[index]
            X_data = np.delete(X_data, index, axis=0)
            y_data = np.delete(y_data, index, axis=0)
            
        return [X_train_set, X_val_set, X_data, y_train_set, y_val_set, y_data]   


# Normalize data
def normalize(X):
    # Find the min and max values for each column
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    # Normalize
    for x in X:
        for j in range(X.shape[1]):
            x[j] = (x[j]-x_min[j])/(x_max[j]-x_min[j])




    

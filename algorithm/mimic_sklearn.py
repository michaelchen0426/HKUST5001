from sklearn import datasets
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd

#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn import datasets
from sklearn.linear_model import SGDClassifier

# Create dataset of classification task with many redundant and few
# informative features
from datetime import datetime

# Keras: https://keras.io 

class MimicSKLearn(object):
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data

    # 'n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale' is to generate the problem
    # 'penalty','l1_ratio','alpha','max_iter','random_state','n_jobs' is to build the classifier

    # 1. change 'penalty' to number
    # 2. Draw plot to see the trend for each attributes. (may ignore)
    # 3. Calculate covariance. 
    # 4. Apply 
    #   - regression directly? https://www.tensorflow.org/tutorials/keras/basic_regression 
    #   - PCA? 
    #   - Neural Network?
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def mimic_process(self):
        my_times = []

        for index, record in self.training_data.iterrows():
            startTime = datetime.now()
            X, Y = datasets.make_classification(
                n_samples=record['n_samples'], 
                n_features=record['n_features'],
                n_classes=record['n_classes'],
                n_clusters_per_class=record['n_clusters_per_class'],
                n_informative=record['n_informative'], 
                flip_y=record['flip_y'],
                scale=record['scale'],
                )
            
            clf = SGDClassifier(
                penalty=record['penalty'],
                l1_ratio=record['l1_ratio'],
                alpha=record['alpha'],
                max_iter=record['max_iter'],
                random_state=record['random_state'],
                n_jobs=record['n_jobs']
            )
            clf.fit(X, Y)
            seconds = (datetime.now() - startTime).total_seconds()
            my_times.append(seconds)
            print('-----', index, '-----: ', seconds)
        
        df = pd.DataFrame(data={"my_time": my_times})

        df.to_csv('my_times_all.csv', sep='\t')
    
    def mimic_test_process(self):
        my_times = []

        for index, record in self.testing_data.iterrows():
            startTime = datetime.now()
            X, Y = datasets.make_classification(
                n_samples=record['n_samples'], 
                n_features=record['n_features'],
                n_classes=record['n_classes'],
                n_clusters_per_class=record['n_clusters_per_class'],
                n_informative=record['n_informative'], 
                flip_y=record['flip_y'],
                scale=record['scale'],
                )
            
            clf = SGDClassifier(
                penalty=record['penalty'],
                l1_ratio=record['l1_ratio'],
                alpha=record['alpha'],
                max_iter=record['max_iter'],
                random_state=record['random_state'],
                n_jobs=record['n_jobs']
            )
            clf.fit(X, Y)
            seconds = (datetime.now() - startTime).total_seconds()
            my_times.append(seconds)
            print('-----', index, '-----: ', seconds)
        
        df = pd.DataFrame(data={"my_time": my_times})

        df.to_csv('my_times_all_testing.csv', sep='\t')
        

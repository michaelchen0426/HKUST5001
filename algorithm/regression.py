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

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

n_job_minus_1 = 32
dense_num = 128

# Keras: https://keras.io 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class SKLearnAlgorithm(object):
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        self.EPOCHS = 50000
        self.sample_size = 380

    # 'n_samples','n_features','n_classes','n_clusters_per_class','n_informative','flip_y','scale' is to generate the problem
    # 'penalty','l1_ratio','alpha','max_iter','random_state','n_jobs' is to build the classifier

    # 1. change 'penalty' to number
    # 2. Change n_job_minus_1 = 32
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def transfer_penalty(self, data):
        new_penalty = []

        for item in data:
            if item == 'none':
                new_penalty.append(0)
            elif item == 'l1':
                new_penalty.append(1)
            elif item == 'l2':
                new_penalty.append(2)
            elif item == 'elasticnet':
                new_penalty.append(3)
            else:
                new_penalty.append(-1)
        
        return new_penalty
    
    def transfer_n_jobs(self, data):
        new_jobs = []

        for item in data:
            if item == -1:
                new_jobs.append(n_job_minus_1)
            else:
                new_jobs.append(item)
        
        return new_jobs

    def transform_data_with_validation(self):
        # Get Training_label
        self.training_label = self.training_data['time']

        # Transfer Penalty value to numeric for training data
        self.training_data['penalty'] = self.transfer_penalty(self.training_data['penalty'])
        self.training_data['n_jobs'] = self.transfer_n_jobs(self.training_data['n_jobs'])

        # Transfer Penalty value to numeric for testing data
        self.testing_data['penalty'] = self.transfer_penalty(self.testing_data['penalty'])
        self.testing_data['n_jobs'] = self.transfer_n_jobs(self.testing_data['n_jobs'])

        # Remove id and time column
        self.training_data.drop(['id'], axis=1, inplace=True)
        self.testing_data.drop(['id'], axis=1, inplace=True)
        self.training_data.drop(['time'], axis=1, inplace=True)

        # Split Training data into two part, training and validation
        self.training_data, self.validation_data = np.split(self.training_data, [self.sample_size], axis = 0)
        self.training_label, self.validation_label = np.split(self.training_label, [self.sample_size], axis = 0)

        # Normalize features
        mean = self.training_data.mean(axis=0)
        std = self.training_data.std(axis=0)
        self.training_data = (self.training_data - mean) / std
        self.testing_data = (self.testing_data - mean) / std
        self.validation_data = (self.validation_data - mean) / std
        
        # print("Real Training Data:")
        # print(self.training_data)
        
        # print("Sample  Data:")
        # print(self.validation_data)
    
    def transform_data_without_validation(self):
        # Get Training_label
        self.training_label = self.training_data['time']

        # Transfer Penalty value to numeric for training data
        self.training_data['penalty'] = self.transfer_penalty(self.training_data['penalty'])
        self.training_data['n_jobs'] = self.transfer_n_jobs(self.training_data['n_jobs'])

        # Transfer Penalty value to numeric for testing data
        self.testing_data['penalty'] = self.transfer_penalty(self.testing_data['penalty'])
        self.testing_data['n_jobs'] = self.transfer_n_jobs(self.testing_data['n_jobs'])

        # Remove id and time column
        self.training_data.drop(['id'], axis=1, inplace=True)
        self.testing_data.drop(['id'], axis=1, inplace=True)
        self.training_data.drop(['time'], axis=1, inplace=True)

        # Normalize features
        mean = self.training_data.mean(axis=0)
        std = self.training_data.std(axis=0)
        self.training_data = (self.training_data - mean) / std
        self.testing_data = (self.testing_data - mean) / std
        
        print("Transformed Training Data:")
        print(self.training_data)

        print("Transformed Testing Data:")
        print(self.testing_data)

    def build_model(self, data):
        print('--build_model--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            #0.01 has 2.xx point with 3 layers of 128 units 
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.005), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.005), activation=tf.nn.relu),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.005), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)
        #optimizer = tf.train.GradientDescentOptimizer(0.005)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse', 'mae'])
        return model
    
    def build_model_2(self, data):
        print('--build_model_2--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse'])
        return model

    def build_model_3(self, data):
        print('--build_model_3--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.01), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.01), activation=tf.nn.relu),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.01), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)
        #optimizer = tf.train.GradientDescentOptimizer(0.005)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse'])
        return model
    
    def build_model_4(self, data):
        print('--build_model_4--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num/2,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu),
            keras.layers.Dense(dense_num/4,  kernel_regularizer=keras.regularizers.l1(0.003), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse'])
        return model
    
    def build_model_5(self, data):
        print('--build_model_5--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse'])
        return model
    
    def build_model_6(self, data):
        print('--build_model_6--:', data.shape[1])
        model = keras.Sequential([
            #, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
            keras.layers.Dense(dense_num,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(dense_num/2,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
            keras.layers.Dense(dense_num/4,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mse'])
        return model

    # def build_model_2(self, data):
    #     model = keras.Sequential([
    #         keras.layers.Dense(64, activation=tf.nn.relu,
    #                         input_shape=(data.shape[1],)),
    #         keras.layers.Dense(16, activation=tf.nn.relu),
    #         keras.layers.Dense(1)
    #     ])

    #     optimizer = tf.train.RMSPropOptimizer(0.001)

    #     model.compile(loss='mse',
    #                     optimizer=optimizer,
    #                     metrics=['mse'])
    #     return model

    def plot_history(self, history):
        # Has bug with matplotlib in OSX: Working with Matplotlib on OSX

        # plt.figure()
        # plt.xlabel('Epoch')
        # plt.ylabel('Mean Abs Error [1000$]')
        # plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
        #         label='Train Loss')
        # plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
        #         label = 'Val loss')
        # plt.legend(history.history['mean_absolute_error'])
        # plt.ylim([0, 5])
        print('Train Loss')
        # print(history.history['mean_absolute_error'])
        print('Val loss')
        #print(history.history['val_mean_absolute_error'])

    def handle_negative_value(self, result):
        new_result = []

        for data in result:
            if data >= 0:
                new_result.append(data)
            else:
                new_result.append(0)
                #new_result.append(-data)
        
        return new_result

    def output_result(self, result):
        new_result = self.handle_negative_value(result)

        df = pd.DataFrame(data={'Time': new_result})
        df.rename_axis("Id", axis="columns")
        df.to_csv('result.csv', sep='\t')

    def run_with_validation(self):
        # transfer the training data

        self.transform_data_with_validation()

        # train the model
        model = self.build_model(self.training_data)
        model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        # early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta = 0, 
            patience=200
        )

        history = model.fit(self.training_data, self.training_label, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        #history = model.fit(self.training_data, self.training_label, epochs=self.EPOCHS,
        #                    validation_split=0.1, verbose=0)

        self.plot_history(history)

        # Evaluate by testing data
        test_predictions = model.predict(self.validation_data).flatten()

        combined_df = pd.DataFrame(dict(predicted=test_predictions, real=self.validation_label))
        print("Result:")
        print(combined_df)

        rms = np.sqrt(mean_squared_error(self.validation_label, test_predictions))
        print("MSE:")
        print(rms)

    def run_without_validation(self):
        # transfer the training data

        self.transform_data_without_validation()

        # train the model
        model = self.build_model(self.training_data)
        model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        # early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)
        early_stop = keras.callbacks.EarlyStopping(
            #monitor='loss', 
            monitor='val_mean_squared_error',
            min_delta = 0, 
            patience=500
        )

        history = model.fit(self.training_data, self.training_label, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=2, shuffle=True,
                            callbacks=[early_stop, PrintDot()])

        self.plot_history(history)

        # Evaluate by testing data
        test_predictions = model.predict(self.testing_data).flatten()

        print("Result:")
        print(test_predictions)

        self.output_result(test_predictions)

    def feature_ranking(self):
        self.transform_data_without_validation()
        
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=5)
        selector = selector.fit(self.training_data, self.training_label)
        print(selector.support_)
        print(selector.ranking_)
    # def draw_plot(self):

    # def generate_problem(self):
    #     self.training_data['n_samples'][0]
    #     n_features
    #     n_classes
    #     n_clusters_per_class
    #     n_informative
    #     flip_y
    #     scale
    #     X, y = datasets.make_classification(n_samples=100000, n_features=20,
    #                                 n_informative=2, n_redundant=2)
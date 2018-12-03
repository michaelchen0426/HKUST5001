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

n_job_minus_1 = 32

# Keras: https://keras.io 
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class SeperateModel(object):
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        self.EPOCHS = 500
        self.sample_size = 380

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
            #, kernel_regularizer=keras.regularizers.l2(0.001)
            keras.layers.Dense(64,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(64,  kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae'])
        return model

    def build_model_2(self, data):
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu,
                            input_shape=(data.shape[1],)),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae'])
        return model

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
        print(history.history['mean_absolute_error'])
        print('Val loss')
        print(history.history['val_mean_absolute_error'])

    def handle_negative_value(self, result):
        new_result = []

        for data in result:
            if data >= 0:
                new_result.append(data)
            else:
                new_result.append(-data)
        
        return new_result

    def output_result(self, result):
        new_result = self.handle_negative_value(result)

        df = pd.DataFrame(data={'time': new_result})

        df.to_csv('result_seperate_model.csv', sep='\t')

    def build_model_for_each_penalty(self):
        # self.training_data.drop(['id'], axis=1, inplace=True)
        # self.training_data['penalty'] = self.training_data['penalty'].replace(to_replace=['none', 'l1', 'l2', 'elasticnet'], value=[0, 1, 2, 3])

        # Transfer Penalty value to numeric for training data
        self.training_data['penalty'] = self.transfer_penalty(self.training_data['penalty'])
        self.training_data['n_jobs'] = self.transfer_n_jobs(self.training_data['n_jobs'])

        # Transfer Penalty value to numeric for testing data
        self.testing_data['penalty'] = self.transfer_penalty(self.testing_data['penalty'])
        self.testing_data['n_jobs'] = self.transfer_n_jobs(self.testing_data['n_jobs'])

        df1 = self.training_data[self.training_data['penalty'] == 0]
        df2 = self.training_data[self.training_data['penalty'] == 1]
        df3 = self.training_data[self.training_data['penalty'] == 2]
        df4 = self.training_data[self.training_data['penalty'] == 3]

        label1 = df1['time']
        df1.drop(['time'], axis=1, inplace=True)
        df1.drop(['penalty'], axis=1, inplace=True)
        label2 = df2['time']
        df2.drop(['time'], axis=1, inplace=True)
        df2.drop(['penalty'], axis=1, inplace=True)
        label3 = df3['time']
        df3.drop(['time'], axis=1, inplace=True)
        df3.drop(['penalty'], axis=1, inplace=True)
        label4 = df4['time']
        df4.drop(['time'], axis=1, inplace=True)
        df4.drop(['penalty'], axis=1, inplace=True)

        # Split Training data into two part, training and validation
        # # Train for Penalty 0 - Good result
        X_train1, X_test1, y_train1, y_test1 = train_test_split(df1, label1)
        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train1)

        StandardScaler(copy=True, with_mean=True, with_std=True)

        # Now apply the transformations to the data:
        X_train1 = scaler.transform(X_train1)
        X_test1 = scaler.transform(X_test1)

        # train the model
        model1 = self.build_model(X_train1)
        #model1.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

        model1.fit(X_train1, y_train1, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Evaluate by testing data
        test_predictions1 = model1.predict(X_test1).flatten()

        rms1 = np.sqrt(mean_squared_error(y_test1, test_predictions1))
        print("MSE penalty0:", rms1)

        # Train for Penalty 1
        X_train2, X_test2, y_train2, y_test2 = train_test_split(df2, label2)
        scaler2 = StandardScaler()
        # Fit only to the training data
        scaler2.fit(X_train2)

        StandardScaler(copy=True, with_mean=True, with_std=True)

        # Now apply the transformations to the data:
        X_train2 = scaler2.transform(X_train2)
        X_test2 = scaler2.transform(X_test2)

        # train the model
        model2 = self.build_model(X_train2)
        #model2.summary()

        model2.fit(X_train2, y_train2, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Evaluate by testing data
        test_predictions2 = model2.predict(X_test2).flatten()

        rms2 = np.sqrt(mean_squared_error(y_test2, test_predictions2))
        print("MSE penalty1:", rms2)

        # Train for Penalty 2  - Good result
        X_train3, X_test3, y_train3, y_test3 = train_test_split(df3, label3)
        scaler3 = StandardScaler()
        # Fit only to the training data
        scaler3.fit(X_train3)

        StandardScaler(copy=True, with_mean=True, with_std=True)

        # Now apply the transformations to the data:
        X_train3 = scaler3.transform(X_train3)
        X_test3 = scaler3.transform(X_test3)

        # train the model
        model3 = self.build_model(X_train2)
        #model3.summary()

        model3.fit(X_train3, y_train3, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Evaluate by testing data
        test_predictions3 = model3.predict(X_test3).flatten()

        rms3 = np.sqrt(mean_squared_error(y_test3, test_predictions3))
        print("MSE penalty2:", rms3)

        # Train for Penalty 3
        X_train4, X_test4, y_train4, y_test4 = train_test_split(df4, label4)
        scaler4 = StandardScaler()
        # Fit only to the training data
        scaler4.fit(X_train4)

        StandardScaler(copy=True, with_mean=True, with_std=True)

        # Now apply the transformations to the data:
        X_train4 = scaler4.transform(X_train4)
        X_test4 = scaler4.transform(X_test4)

        # train the model
        model4 = self.build_model(X_train4)
        #model4.summary()

        model4.fit(X_train4, y_train4, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Evaluate by testing data
        test_predictions4 = model4.predict(X_test4).flatten()

        rms4 = np.sqrt(mean_squared_error(y_test4, test_predictions4))
        print("MSE penalty3:", rms4)

        # Predict testing data:

        original_testing_data = self.testing_data

        no_id_testing_data = self.testing_data.drop(['id'], axis=1)

        tf1 = no_id_testing_data[no_id_testing_data['penalty'] == 0]
        tfx1 = original_testing_data[original_testing_data['penalty'] == 0]
        tf2 = no_id_testing_data[no_id_testing_data['penalty'] == 1]
        tfx2 = original_testing_data[original_testing_data['penalty'] == 1]
        tf3 = no_id_testing_data[no_id_testing_data['penalty'] == 2]
        tfx3 = original_testing_data[original_testing_data['penalty'] == 2]
        tf4 = no_id_testing_data[no_id_testing_data['penalty'] == 3]
        tfx4 = original_testing_data[original_testing_data['penalty'] == 3]

        tf1.drop(['penalty'], axis=1, inplace=True)
        tf2.drop(['penalty'], axis=1, inplace=True)
        tf3.drop(['penalty'], axis=1, inplace=True)
        tf4.drop(['penalty'], axis=1, inplace=True)

        
        tf1 = scaler.transform(tf1)
        tf2 = scaler2.transform(tf2)
        tf3 = scaler3.transform(tf3)
        tf4 = scaler4.transform(tf4)

        new_prediction1 = model1.predict(tf1).flatten()
        new_prediction2 = model2.predict(tf2).flatten()
        new_prediction3 = model3.predict(tf3).flatten()
        new_prediction4 = model4.predict(tf4).flatten()

        print(new_prediction1)

        combined_df1 = pd.DataFrame(dict(training=tfx1['id'], predicts=new_prediction1))
        combined_df1.to_csv('cl1.csv', sep='\t')
        combined_df2 = pd.DataFrame(dict(training=tfx2['id'], predicts=new_prediction2))
        combined_df2.to_csv('cl2.csv', sep='\t')
        combined_df3 = pd.DataFrame(dict(training=tfx3['id'], predicts=new_prediction3))
        combined_df3.to_csv('cl3.csv', sep='\t')
        combined_df4 = pd.DataFrame(dict(training=tfx4['id'], predicts=new_prediction4))
        combined_df4.to_csv('cl4.csv', sep='\t')
    

    def build_model_for_each_penalty_skera(self):
        self.training_data.drop(['id'], axis=1, inplace=True)
        # self.training_data['penalty'] = self.training_data['penalty'].replace(to_replace=['none', 'l1', 'l2', 'elasticnet'], value=[0, 1, 2, 3])

        # Transfer Penalty value to numeric for training data
        self.training_data['penalty'] = self.transfer_penalty(self.training_data['penalty'])
        self.training_data['n_jobs'] = self.transfer_n_jobs(self.training_data['n_jobs'])

        # Transfer Penalty value to numeric for testing data
        self.testing_data['penalty'] = self.transfer_penalty(self.testing_data['penalty'])
        self.testing_data['n_jobs'] = self.transfer_n_jobs(self.testing_data['n_jobs'])

        df1 = self.training_data[self.training_data['penalty'] == 0]
        df2 = self.training_data[self.training_data['penalty'] == 1]
        df3 = self.training_data[self.training_data['penalty'] == 2]
        df4 = self.training_data[self.training_data['penalty'] == 3]
        print('-----0-----')
        print(df1.shape)

        label1 = df1['time']
        df1.drop(['time'], axis=1, inplace=True)
        df1.drop(['penalty'], axis=1, inplace=True)
        label2 = df2['time']
        df2.drop(['time'], axis=1, inplace=True)
        df2.drop(['penalty'], axis=1, inplace=True)
        label3 = df3['time']
        df3.drop(['time'], axis=1, inplace=True)
        df3.drop(['penalty'], axis=1, inplace=True)
        label4 = df4['time']
        df4.drop(['time'], axis=1, inplace=True)
        df4.drop(['penalty'], axis=1, inplace=True)

        # Normalize features
        mean1 = df1.mean(axis=0)
        std1 = df1.std(axis=0)
        df1 = (df1 - mean1) / std1

        mean2 = df2.mean(axis=0)
        std2 = df2.std(axis=0)
        df2 = (df2 - mean2) / std2

        mean3 = df3.mean(axis=0)
        std3 = df3.std(axis=0)
        df3 = (df3 - mean3) / std3

        mean4 = df4.mean(axis=0)
        std4 = df4.std(axis=0)
        df4 = (df4 - mean4) / std4


        # Split Training data into two part, training and validation
        # Train for Penalty 0 - Good result
    
        # train the model
        model1 = self.build_model(df1)
        #model1.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

        model1.fit(df1, label1, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Train for Penalty 1
        
        # train the model
        model2 = self.build_model(df2)
        #model2.summary()

        model2.fit(df2, label2, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Train for Penalty 2  - Good result
        # train the model
        model3 = self.build_model(df3)
        #model3.summary()

        model3.fit(df3, label3, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        
        # Train for Penalty 3
        # train the model
        model4 = self.build_model(df4)
        #model4.summary()

        model4.fit(df4, label4, epochs=self.EPOCHS,
                            validation_split=0.1, verbose=0,
                            callbacks=[early_stop, PrintDot()])

        # Predict testing data:
        print('----1-----')
        original_testing_data = self.testing_data
        print(original_testing_data.loc[[1]])
        print(original_testing_data.shape)

        print('----2-----')
        no_id_testing_data = self.testing_data.drop(['id'], axis=1)
        print(no_id_testing_data.loc[[1]])
        print(no_id_testing_data.shape)

        tf1 = no_id_testing_data[no_id_testing_data['penalty'] == 0]
        tfx1 = original_testing_data[original_testing_data['penalty'] == 0]
        tf2 = no_id_testing_data[no_id_testing_data['penalty'] == 1]
        tfx2 = original_testing_data[original_testing_data['penalty'] == 1]
        tf3 = no_id_testing_data[no_id_testing_data['penalty'] == 2]
        tfx3 = original_testing_data[original_testing_data['penalty'] == 2]
        tf4 = no_id_testing_data[no_id_testing_data['penalty'] == 3]
        tfx4 = original_testing_data[original_testing_data['penalty'] == 3]

        tf1.drop(['penalty'], axis=1, inplace=True)
        tf2.drop(['penalty'], axis=1, inplace=True)
        tf3.drop(['penalty'], axis=1, inplace=True)
        tf4.drop(['penalty'], axis=1, inplace=True)

        tf1 = (tf1 - mean1) / std1
        tf2 = (tf2 - mean2) / std2
        tf3 = (tf3 - mean3) / std3
        tf4 = (tf4 - mean4) / std4

        new_prediction1 = model1.predict(tf1).flatten()
        new_prediction2 = model2.predict(tf2).flatten()
        new_prediction3 = model3.predict(tf3).flatten()
        new_prediction4 = model4.predict(tf4).flatten()

        combined_df1 = pd.DataFrame(dict(training=tfx1['id'], predicts=new_prediction1))
        combined_df1.to_csv('cl1.csv', sep='\t')
        combined_df2 = pd.DataFrame(dict(training=tfx2['id'], predicts=new_prediction2))
        combined_df2.to_csv('cl2.csv', sep='\t')
        combined_df3 = pd.DataFrame(dict(training=tfx3['id'], predicts=new_prediction3))
        combined_df3.to_csv('cl3.csv', sep='\t')
        combined_df4 = pd.DataFrame(dict(training=tfx4['id'], predicts=new_prediction4))
        combined_df4.to_csv('cl4.csv', sep='\t')



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
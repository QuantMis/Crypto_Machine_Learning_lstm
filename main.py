import numpy as np 
import plotly.offline as py 
import plotly.graph_objs as go 

# keras
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 

import json
import os  

# auxilliary 
from helpers.generate_labels import GenLabels  
from helpers.poly_interpolation import PolyInter  

# indicators as Input
from derivative.rsi import StochRsi
from derivative.macd import Macd

class genLSTM:

    def __init__(self):
        # set params
        timesteps = 10
        epochs = 10
        batch_size = 8
        self.loss = "categorical_crossentropy"
        self.optimizer = "adam"
        self.metric = ['accuracy']
        model_tag = "lstm_model.h5"

        # load and reshape data
        with open('filename.json') as f:
            data = json.load(f)
        X, y = self.extract_data(np.array(data['close']))
        X, y = self.shape_data(X, y, timesteps)

        # ensure equal number of labels, shuffle and split
        X_train, X_test, y_train, y_test = self.adjust_data(X, y)
        
        # binary encode for softmax function
        y_train, y_test = to_categorical(y_train, 2), to_categorical(y_train, 2)

        # build & train model
        model = self.build_model()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test))
        model.save(f'models/{model_tag}')

    
    def extract_data(self, data):
        # obtain labels
        labels = GenLabels(data, window=25, polyorder=3).labels

        # obtain features
        macd = Macd(data, 6, 12, 3).values()
        stoch_rsi = StochRsi(data, period=14).hist_values
        inter_slope = PolyInter(data, progress_bar=True).values
        
        # truncate bad values and shift label
        X = np.array([macd[30:-1],
            stoch_rsi[30:-1],
            inter_slope[30:-1]
        ])

        X = np.transpose(X)
        labels = labels[31:]

        return X, labels

    
    
    def shape_data(self, X, y, timesteps):
        # scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, 'models/scaler.dump')

        # reshape data with timesteps
        reshaped = []
        for i in range(timesteps, X.shape[0] + 1):
            reshaped.append(X[i-timesteps:i])
        
        # account for data lost in reshaping
        X = np.array(reshaped)
        y = y[timesteps - 1:]

        return X, y

    
    def adjust_data(self, X, y, split=0.8):
        # count n for each label
        count_1 = np.count_nonzero(y)
        count_0 = y.shape[0] - count_1
        cut = min(count_0, count_1)

        # split data by split ratio for testing
        train_idx = int(cut * split)

        # shuffle data
        np.random.seed(42)
        shuffle_index = np.random.permutation(X.shape[0])
        X,y = X[shuffle_index], y[shuffle_index]

        # find indexes of each label
        idx_1 = np.argwhere(y==1).flatten()
        idx_0 = np.argwhere(y==0).flatten()

        # grab specified cut of each label and put them together
        X_train = np.concatenate((X[idx_1[:train_idx]], X[idx_0[:train_idx]]), axis=0)
        X_test = np.concatenate((X[idx_1[train_idx:cut]], X[idx_0[train_idx:cut]]), axis=0)
        y_train = np.concatenate((y[idx_1[:train_idx]], y[idx_0[:train_idx]]), axis=0)
        y_test = np.concatenate((y[idx_1[train_idx:cut]], y[idx_0[train_idx:cut]]), axis=0)
        
        # shuffle again to mix labels
        np.random.seed(7)
        shuffle_train = np.random.permutation(X_train.shape[0])
        shuffle_test = np.random.permutation(X_test.shape[0])

        X_train, y_train = X_train[shuffle_train], y_train[shuffle_train]
        X_test, y_test = X_test[shuffle_test], y_test[shuffle_test]

        return X_train, X_test, y_train, y_test
    

    def build_model(self):
        model = Sequential()

        # first layer
        model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))

        # second layer
        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))

        # third later and output
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # compile layer
        model.compile(loss=self.loss, optimizer=self.optimizer, metric=self.metric)
        return model
    

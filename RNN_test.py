# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:17:24 2019

@author: Thilina Weerathunga
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
from my_answers import *

#%%
dataset = pd.read_csv('datasets/qqq_prices.csv',header=None)

#%%

from sklearn.preprocessing import MinMaxScaler
dataset.head()
scaler = MinMaxScaler()
scaler.fit(dataset)

#%%
print(scaler.data_max_)
print(scaler.data_min_)

#%%
normalized_dataset = scaler.transform(dataset)

#%%
#plt.plot(normalized_dataset,"g-")
#plt.xlabel('time period')
#plt.ylabel('normalized series value')
#plt.grid(which='minor', alpha=0.2)
#plt.grid(which='major', alpha=0.5)

#%%
from my_answers import window_transform_series

window_size = 5
X,y = window_transform_series(series = normalized_dataset,window_size = window_size)

#%%
print(X[0:5])
print(y[0:5])

#%%
# split our dataset into training / testing sets
train_test_split = int(np.ceil(2*len(y)/float(3)))   # set the split point

# partition the training set
X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

# keep the last chunk for testing
X_test = X[train_test_split:,:]
y_test = y[train_test_split:]

# NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
X_train_new = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test_new = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

#%%
### Done: create required RNN model
# import keras network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
np.random.seed(0)


# TODO: implement build_part1_RNN in my_answers.py
from my_answers import build_part1_RNN
model = build_part1_RNN(window_size)

# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

#%%
# run your model!
model.fit(X_train_new, y_train, epochs=1000, batch_size=50, verbose=0)

#%%
# generate predictions for training
train_predict = model.predict(X_train_new)
test_predict = model.predict(X_test_new)

#%%
# print out training and testing errors
training_error = model.evaluate(X_train_new, y_train, verbose=0)
print('training error = ' + str(training_error))

testing_error = model.evaluate(X_test_new, y_test, verbose=0)
print('testing error = ' + str(testing_error))

#%%
### Plot everything - the original series as well as predictions on training and testing sets
#import datetime as dt

#dates        = pd.read_csv('datasets/dates.csv',header=None,names=['Date'])
#dates_frmted = dates.apply(lambda x : dt.datetime.strptime(x.values[0].strip('00:00:00').strip(' '),'%Y-%m-%d').date(),axis=1)

#x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in dates_frmted]
#y = range(len(x)) # many thanks to Kyss Tao for setting me straight here

x_array = [i for i in range(0,166) ] 


#%%
# plot original series
#plt.plot(normalized_dataset,color = 'g')
plt.plot(x_array,scaler.inverse_transform(normalized_dataset),color = 'g')
plt.xticks(x_array,rotation=45)
plt.grid(which='both')

# plot training set prediction
split_pt = train_test_split + window_size 
plt.plot(np.arange(window_size,split_pt,1),scaler.inverse_transform(train_predict),color = 'b')

# plot testing set prediction
plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),scaler.inverse_transform(test_predict),color = 'r')

# pretty up graph
plt.xlabel('day')
plt.ylabel('Price of QQQ stock')
plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.show()

#%%




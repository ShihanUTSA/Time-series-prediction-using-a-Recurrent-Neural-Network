import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import string


# Done: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
''' ******************************************************** '''
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X,y = [],[]
        
    for idx in range(len(series)-window_size):
        X.append(series[idx:idx+window_size])
    
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    
    return X,y


''' ******************************************************** '''
# Done: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    RNN_model = Sequential()
    RNN_model.add(LSTM(5, input_shape=(window_size, 1)))
    RNN_model.add(Dense(1))
    return RNN_model
    #pass


''' ******************************************************** '''
### Done: return the text input with only ascii lowercase and the punctuation 
#   given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    extra1=['\xa0', '¢', '¨', '©', 'ã']
    extra2=['à', 'â', 'è', 'é']
    remove_set=(set(string.printable) | set(extra1) |set(extra2))-(set(string.ascii_lowercase)| set(punctuation) |set(' '))
    remove_set=list(remove_set)
    for c in remove_set:
        text=text.replace(c,' ')
    
    return text

''' ******************************************************** '''
### Done: fill out the function below that transforms the input text and 
##  window-size into a set of input/output pairs for use with our RNN model

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    idx = 0
    
    while idx<(len(text)-window_size):
        inputs.append(text[idx:idx+window_size])
        outputs.append(text[idx+window_size])
        idx += step_size
    
      
    # reshape each 
    '''
    inputs = np.asarray(inputs)
    inputs.shape = (len(inputs),1)
    outputs = np.asarray(outputs)
    outputs.shape = (len(outputs),1)
    '''
    
    return inputs,outputs

''' ******************************************************** '''
# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
    

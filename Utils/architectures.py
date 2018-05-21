
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense
from keras.layers import Flatten, BatchNormalization, Activation, MaxPool2D
from keras.layers import Conv1D
from keras import backend as K


def create_iNet():
    '''Proposed modifications to Nvidia PilotNet model. 

    '''
    model = Sequential()
    model.add(Conv2D(64, 3, data_format='channels_last', kernel_initializer='he_normal',
                     input_shape=(160, 320, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Dropout(0.6))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.6))

    model.add(Dense(1))

    model.summary()

    return model


def build_model(input_shape=(160,320,3), dropout=0.6):
    """
   Classic NVIDIA model.

    """
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(BatchNormalization())
    
    model.add(Dropout(dropout))
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def build_model2(input_shape=(160,320,3), dropout=0.6):
    """
    Modified NVIDIA model. Conv1D classificator.

    """
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))


    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))


    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

   
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.6))
    
    model.add(Dropout(dropout))
    model.add(Flatten())
    
    model.add(Conv1D(100, (3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Conv1D(50, (3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Conv1D(10, (3)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Dense(1))
    model.summary()

    return model


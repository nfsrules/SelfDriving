import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from architectures import *


#Load data
X = np.load('Dataset/train_X.npy')
Y = np.load('Dataset/train_Y.npy')
Y = Y.astype(np.float32)
print('Training X shape =', np.shape(X))
print('Training Y shape =', np.shape(Y))


# Create model
model = create_iNet()


# Custon RMSE loss
#def root_mean_squared_error(y_true, y_pred):
#        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics =["accuracy"]
             )


#checkpointer = ModelCheckpoint(filepath='autopilot_checkpoint_mse.h5', verbose=1, save_best_only=True)

reduce_LR  = ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                               patience=2,verbose=True)

early_stopping_monitor = EarlyStopping(patience=3, monitor='val_loss', mode='auto')


history = model.fit(X, Y, 
                    batch_size=100, 
                    epochs=30, 
                    validation_split=0.15,
                    callbacks=[reduce_LR, early_stopping_monitor],
                    shuffle=True)


model.save('autopilot_nvidia.hdf5')


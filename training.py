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

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15, random_state=53)
print('Training X shape =', np.shape(train_X))
print('Training Y shape =', np.shape(train_Y))

# Create model from architectures
model = create_iNet()

# Custon RMSE loss
#def root_mean_squared_error(y_true, y_pred):
#        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics =["accuracy"]
             )

# Usefull Keras Callbacks
#checkpointer = ModelCheckpoint(filepath='autopilot_checkpoint_mse.h5', verbose=1, save_best_only=True)

# Learning rate scheduler
reduce_LR  = ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                               patience=2,verbose=True)
# Early stopping monitor
early_stopping_monitor = EarlyStopping(patience=3, monitor='val_loss', mode='auto')

# Train model
history = model.fit(train_X, train_Y, 
                    batch_size=100, 
                    epochs=30, 
                    validation_split=0.15,
                    callbacks=[reduce_LR, early_stopping_monitor],
                    shuffle=True)

# Save model
model.save('autopilot_nvidia.hdf5')


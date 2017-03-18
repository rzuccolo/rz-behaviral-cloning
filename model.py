from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.utils.visualize_util import plot
from keras.callbacks import Callback
import keras.backend as K
import gc


import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sms
import sklearn

import preprocess_util as preprocess


#hyperparameters
INPUT_SHAPE = (66, 200, 3) # in case you change this, it will be necessary to update preprocess_util.py and drive.py as well.
BATCH_SIZE = 256
EPOCHS = 5
EARLY_STOP = 0.02

# code created by Mez Gebre
# https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/
class CustomEarlyStop(Callback):
    """
    Custom Callback that stops the epoch when val_loss reachs user specified value
    This callback assumes you are logging val_loss
    """
    def __init__(self, monitor='val_loss'):
        super(CustomEarlyStop, self).__init__()
        self.monitor = monitor


    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss <= EARLY_STOP:
            print("\nEarly Stop on Epoch {0} with Val_loss {1}\n".format(epoch,val_loss))
            self.model.stop_training = True

# NVIDIA NN            
def my_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24,5,5,activation='relu', subsample=(2, 2), name='conv1-5x5'))
    model.add(Convolution2D(36,5,5,activation='relu', subsample=(2, 2), name='conv2-5x5'))
    model.add(Convolution2D(48,5,5,activation='relu', subsample=(2, 2), name='conv3-5x5'))
    model.add(Convolution2D(64,3,3,activation='relu', subsample=(1, 1), name='conv4-3x3'))
    model.add(Convolution2D(64,3,3,activation='relu', subsample=(1, 1), name='conv5-3x3'))
    model.add(Flatten())
    model.add(Dense(1164, activation='linear'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='linear'))
    model.add(Dropout(.2))
    model.add(Dense(50, activation='linear'))
    model.add(Dropout(.1))
    model.add(Dense(10, activation='linear'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model

            
if __name__ == '__main__':
    
    # Create model
    model = my_nvidia()
    # model.summary()
    
    # Load the image log
    '''
    ATTENTION:
    Load the log here assumes the proper and new log was create based on original driving_log.csv
    Refer to the preprocess_util.py and playground.ipynb notebook for more info. Assumes the new driver log is titeled
    'preprocessed_driver_log.csv'
    '''
    processed_log = preprocess.get_processed_dataframes()
    
    # Split the data for validation
    df_train=processed_log.sample(frac=0.8) # shuffle and keep 80%
    df_validation=processed_log.drop(df_train.index) # 20%

    # Image generator
    def generator_features_and_labels(df, batch_size=32):
        num_samples = df.shape[0]

        while 1: # Loop forever so the generator never terminates
            df = df.sample(frac=1).reset_index(drop=True) # shuffle the rows, reset indexes
            for offset in range(0, num_samples, batch_size):
                batch_samples = df.iloc[offset:offset+batch_size].reset_index(drop=True)

                images = [preprocess.load_image(row) for _, row in batch_samples.iterrows()]
                X_train = np.array(images).reshape((len(images), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
                y_train = batch_samples.steering

                yield sklearn.utils.shuffle(X_train, y_train)
                
    train_generator = generator_features_and_labels(df_train, batch_size=BATCH_SIZE)
    validation_generator = generator_features_and_labels(df_validation, batch_size=BATCH_SIZE)

    # Train the model
    early_stop = CustomEarlyStop(monitor='val_loss')

    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                df_train.shape[0], validation_data=validation_generator, \
                nb_val_samples=df_validation.shape[0], nb_epoch=EPOCHS, verbose=1, callbacks=[early_stop])

    # Print the keys contained in the history object
    #print(history_object.history.keys())

    # Plot the training and validation loss for each epoch
    #plt.plot(history_object.history['loss'])
    #plt.plot(history_object.history['val_loss'])
    #plt.title('model mean squared error loss')
    #plt.ylabel('mean squared error loss')
    #plt.xlabel('epoch')
    #plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()

    model.save("model.h5")
    print("Training complete!")

    #Clean up after ourselves!!
    K.clear_session()
    gc.collect()
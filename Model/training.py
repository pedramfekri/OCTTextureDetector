import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import pandas as pd

import xlrd
import seaborn as sns

from sklearn import preprocessing
import tensorflow as tf

import IPython
import IPython.display


MAX_EPOCHS = 200


def compile_and_fit(model, window, patience=2):
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                   patience=patience,
    #                                                   mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        # callbacks=[early_stopping]
    )
    model.save('../savedModel')
    return history, model


# IPython.display.clear_output()

# RNN
multi_lstm_model = tf.keras.Sequential([

    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    # tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    # tf.keras.layers.Dropout(0.01),
    # Shape => [batch, out_steps*features].
    # tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(OUT_STEPS * 1, activation='relu'),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, 1])
])

history, model = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_window.plot(multi_lstm_model)

'''
# plotting performance in history
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
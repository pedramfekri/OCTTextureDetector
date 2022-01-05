import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def lstm_classification(lstm_unit, dense, time, features, cls, drop=False):
    l = len(lstm_unit)
    l_dense = len(dense)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(time, features)))
    if l == 1:
        model.add(tf.keras.layers.LSTM(lstm_unit[0], return_sequences=False))
    else:
        i = 1
        for layer in lstm_unit:
            if i == l:
                model.add(tf.keras.layers.LSTM(layer, return_sequences=False))
            else:
                model.add(tf.keras.layers.LSTM(layer, return_sequences=True))
            i = i + 1

    if l_dense == 1:
        model.add(tf.keras.layers.Dense(dense[0], activation='relu'))
    else:
        i = 0
        for layer in dense:
            i = i + 1
            if drop and i > 1:
                model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(layer, activation='relu'))

    if drop:
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(cls))

    optimizer = tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)
    # optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # Note: from_logits=True means the output layer does not apply softmax or the outputs are not normal
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.build()
    model.summary()

    return model

'''
if __name__ == '__main__':
    lstm_classification([64, 32, 16], [32, 32, 16, 16], 20, 4, 5, drop=False)
'''
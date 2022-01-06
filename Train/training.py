import tensorflow as tf
from Dataset import dataset as data
from Model import model as mdl
from datetime import datetime
import numpy as np

MAX_EPOCHS = 50


def fit(model, x_train, y_train, x_test, y_test):
    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    checkpoint_path = "SavedModels/checkpoints/"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x=x_train, y=y_train,
                        epochs=MAX_EPOCHS,
                        batch_size=32,
                        validation_data=(x_test, y_test),
                        callbacks=[cp_callback, tensorboard_callback]
                        )
    model.trainable = False
    model.evaluate(x=x_test, y=y_test)
    model.save('../savedModel')
    model.save_weights('SavedModels/Weights/' + str(MAX_EPOCHS) + ".hdf5")
    # model.save_weights('SavedModels/')
    print("sava weight done..")

    model.save('SavedModels/EntireModels/')
    print("sava entire done..")
    model.summary()

    return history, model


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 20
    train, test = data.data_preparation(path)
    x_train, y_train = data.data_windowing(train, time_step)
    x_test, y_test = data.data_windowing(test, time_step)

    print(np.unique(y_train))
    model = mdl.lstm_classification([32], [16, 8], time_step, 4, 3, drop=False)
    # model = mdl.lstm_classification([128, 64], [64, 32, 16, 8], time_step, 4, 3, drop=False)
    # model = mdl.rnn_classification([128], [128, 64, 32, 16], time_step, 4, 3, drop=False)
    fit(model, x_train, y_train, x_test, y_test)
import tensorflow as tf
from Dataset import dataset as data
from Model import model as mdl
from datetime import datetime

MAX_EPOCHS = 200


def fit(model, x, y):
    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    checkpoint_path = "SavedModels/checkpoints/"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x=x, y=y,
                        epochs=MAX_EPOCHS,
                        batch_size=32,
                        callbacks=[cp_callback, tensorboard_callback]
                        )
    model.save('../savedModel')
    model.save_weights('SavedModels/Weights/' + str(MAX_EPOCHS) + ".hdf5")
    # model.save_weights('SavedModels/')
    print("sava weight done..")

    model.save('SavedModels/EntireModels/')
    print("sava entire done..")
    model.summary()
    model.trainable = False
    return history, model


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 20
    train, test = data.data_preparation(path)
    x_train, y_train = data.data_windowing(train, time_step)
    x_test, y_test = data.data_windowing(test, time_step)

    print(y_train)
    # model = mdl.lstm_classification([32], [16], time_step, 4, 5)
    # fit(model, x_train, y_train)
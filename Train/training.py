import tensorflow as tf
from Dataset import dataset as data
from Model import model as mdl
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
MAX_EPOCHS = 15


def fit(model, x_train, y_train, x_test, y_test, x_val, y_val, model_name):
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
                        batch_size=64,
                        validation_data=(x_val, y_val),
                        callbacks=[cp_callback, tensorboard_callback]
                        )
    model.trainable = False
    model.evaluate(x=x_test, y=y_test)
    model.save('../savedModel')
    model.save_weights('SavedModels/Weights/' +model_name+ str(MAX_EPOCHS) + ".hdf5")
    # model.save_weights('SavedModels/')
    print("sava weight done..")

    model.save('SavedModels/EntireModels/')
    print("sava entire done..")
    model.summary()
    json.dump(history.history, open(model_name, 'w'))
    # retrieve
    # history_dict = json.load(open(model_name, 'r'))
    return history, model


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 100
    x_train, y_train, x_test, y_test, x_val, y_val = data.data_preparation(path)
    # x_train, y_train = data.data_windowing(train, time_step)
    # x_test, y_test = data.data_windowing(test, time_step)
    print('train= ', x_train.min(), x_train.max())
    print('test= ', x_test.min(), x_test.max())
    print('val= ', x_val.min(), x_val.max())

    print(np.unique(y_train))
    model = mdl.lstm_classification([32], [16, 8], time_step, 4, 3, drop=False)
    model_name = 'NEWLSTM_[32][16,8]_'
    # model_name = 'LSTM_[32][16,8]_t10_'

    # model = mdl.lstm_classification([8], [8], time_step, 4, 3, drop=False)
    # model_name = 'LSTM_[8][8]_'
    # model_name = 'LSTM_[8][8]_t40_'

    # model = mdl.lstm_classification([32, 32], [32, 16, 8], time_step, 4, 3, drop=False)
    # model_name = 'LSTM[32,32],[32,16,8]_'
    # model_name = 'LSTM[32,32],[32,16,8]_t40_'

    # model = mdl.rnn_classification([32], [16, 8], time_step, 4, 4, drop=False)
    # model_name = 'RNN[32],[16,8]_'
    # model_name = 'NEW_RNN[32],[16,8]_t100_'

    # model = mdl.rnn_classification([32, 32], [32, 16, 8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[32,32],[32,16,8]_'
    # model_name = 'RNN[32,32],[32,16,8]_t40_'

    # model = mdl.rnn_classification([16], [8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[16],[8]_'
    # model_name = 'RNN[16],[8]_t40_'

    # model = mdl.rnn_classification([8], [8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[8],[8]_'
    # model_name = 'RNN[8],[8]_t40_'

    history, model = fit(model, x_train, y_train, x_test, y_test, x_val, y_val, model_name)

    plot_history(history)
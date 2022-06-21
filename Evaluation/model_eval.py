import tensorflow as tf
from Dataset import dataset as data
from Model import model as mdl
from datetime import datetime
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

MAX_EPOCHS = 50


def evl(model, x_test, y_test, model_name):
    model.trainable = False
    checkpoint_path = "../Train/SavedModels/Weights/"
    model.load_weights(checkpoint_path + model_name+'.hdf5')
    print('model was loaded successfully')
    model.evaluate(x=x_test, y=y_test)
    pred_r = model.predict(x_test)
    pred = np.argmax(pred_r, axis=1)
    print(pred)
    confusion = tf.math.confusion_matrix(pred, y_test)
    print(confusion)
    ax = sns.heatmap(confusion)
    plt.show()

    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(y_test, pred, target_names=target_names))


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 20
    train, test = data.data_preparation(path)
    x_train, y_train = data.data_windowing(train, time_step)
    x_test, y_test = data.data_windowing(test, time_step)

    print(np.unique(y_train), np.unique(y_test))
    # model = mdl.lstm_classification([32], [16, 8], time_step, 4, 3, drop=False)

    model = mdl.lstm_classification([32], [16, 8], time_step, 4, 4, drop=False)
    model_name = 'NEWLSTM_[32][16,8]_15'
    # model_name = 'LSTM_[32][16,8]_t10_'

    # model = mdl.lstm_classification([8], [8], time_step, 4, 3, drop=False)
    # model_name = 'LSTM_[8][8]_15'
    # model_name = 'LSTM_[8][8]_t40_'

    # model = mdl.lstm_classification([32, 32], [32, 16, 8], time_step, 4, 3, drop=False)
    # model_name = 'LSTM[32,32],[32,16,8]_15'
    # model_name = 'LSTM[32,32],[32,16,8]_t40_'

    # model = mdl.rnn_classification([32], [16, 8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[32],[16,8]_15'
    # model_name = 'RNN[32],[16,8]_t100_'

    # model = mdl.rnn_classification([32, 32], [32, 16, 8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[32,32],[32,16,8]_15'
    # model_name = 'RNN[32,32],[32,16,8]_t40_'

    # model = mdl.rnn_classification([16], [8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[16],[8]_'
    # model_name = 'RNN[16],[8]_t40_'

    # model = mdl.rnn_classification([8], [8], time_step, 4, 3, drop=False)
    # model_name = 'RNN[8],[8]_'
    # model_name = 'RNN[8],[8]_t40_'

    evl(model, x_test, y_test, model_name)
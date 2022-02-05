import tensorflow as tf
from Dataset import dataset as data
from Model import model as mdl
from datetime import datetime
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

MAX_EPOCHS = 50


def evl(model, x_test, y_test):
    model.trainable = False
    checkpoint_path = "../Train/SavedModels/Weights/"
    model.load_weights(checkpoint_path + '50.hdf5')
    print('model was loaded successfully')
    model.evaluate(x=x_test, y=y_test)
    pred_r = model.predict(x_test)
    pred = np.argmax(pred_r, axis=1)
    print(pred)
    confusion = tf.math.confusion_matrix(pred, y_test)
    print(confusion)
    ax = sns.heatmap(confusion)
    plt.show()


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 20
    train, test = data.data_preparation(path)
    x_train, y_train = data.data_windowing(train, time_step)
    x_test, y_test = data.data_windowing(test, time_step)

    print(np.unique(y_train), np.unique(y_test))
    model = mdl.lstm_classification([32], [16, 8], time_step, 4, 3, drop=False)
    evl(model, x_test, y_test)
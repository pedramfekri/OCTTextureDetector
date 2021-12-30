import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def clean():
    pd.set_option('display.max_columns', None)
    data = pd.read_excel('csv/data.xlsx', index_col=None, header=None)
    print(data.head())
    print(data.shape)
    print(data.columns)
    data = data.drop([0, 1], axis=1)
    data.columns = ['x1', 'y1', 'x2', 'y2', 'label']
    print(data.columns)
    print(data.head())
    data = data.replace('na', 0)
    print(data.head())
    data = data.astype('float32')
    min = 0
    max = 512
    print(data.describe().transpose())

    for col in data.columns:
        if col != 'label':
            # data[col] = (data[col] - data[col].mean()) / data[col].std()
            data[col] = (data[col] - min) / (max - min)
    print(data.describe().transpose())
    data.to_csv('csv/clean.csv', index=False)


def visualization():
    data = pd.read_csv('csv/clean.csv')
    print(data.head(100))
    t = np.arange(0, data.shape[0])
    l = 0
    u = 2000
    # print(type(data['x2'].iloc[100]))

    while True:
        if u >= 25000:
            l = 0
            u = 2000
        plt.clf()
        plt.scatter(t[l: u], data['x2'].iloc[l: u], marker='+')
        plt.scatter(t[l: u], data['x1'].iloc[l: u], marker='+')
        l = l + 150
        u = u + 150
        plt.pause(0.01)

    plt.show()


if __name__ == '__main__':
    visualization()
    # clean()
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


def visualization(data):
    # data = pd.read_csv('csv/clean.csv')
    print(data.head(100))
    t = np.arange(0, data.shape[0])
    l = 0
    u = 2000
    m = data.shape[0]
    # print(type(data['x2'].iloc[100]))

    while True:
        if u >= m:
            l = 0
            u = 2000
        plt.clf()
        plt.scatter(t[l: u], data['x2'].iloc[l: u], marker='+')
        plt.scatter(t[l: u], data['x1'].iloc[l: u], marker='+')
        l = l + 150
        u = u + 150
        plt.pause(0.01)

    plt.show()


def classDivider():
    data = pd.read_csv('csv/clean.csv')
    print('min = ', data['label'].min(), 'max = ', data['label'].max())
    class0 = data.loc[data['label'] == 0]
    class1 = data.loc[data['label'] == 1]
    class2 = data.loc[data['label'] == 2]
    class3 = data.loc[data['label'] == 3]
    class4 = data.loc[data['label'] == 4]

    print('class0 = ', class0.shape, 'class1 = ', class1.shape,
          'class2 = ', class2.shape, 'class3 = ', class3.shape,
          'class4 = ', class4.shape)
    class0.to_csv('csv/class0.csv', index=False)
    class1.to_csv('csv/class1.csv', index=False)
    class2.to_csv('csv/class2.csv', index=False)
    class3.to_csv('csv/class3.csv', index=False)
    class4.to_csv('csv/class4.csv', index=False)
    print('csvs saved!')


if __name__ == '__main__':
    data = pd.read_csv('csv/class0.csv')
    visualization(data)
    # clean()
    # classDivider()
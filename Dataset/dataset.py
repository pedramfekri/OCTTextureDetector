'''
The order of executing the function is as follows:
1- clean()
2- class_divider()
3- data_preparation()
4- data_windowing()

please note that visualization and data_windowing_checker are meant to check the
results
'''
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


def class_divider():
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

    # print(class0.loc[(class0['x1'] == 0) & (class0['x2'] == 0) & (class0['y1'] == 0) & (class0['y2'] == 0)].index)
    class0 = class0.drop(class0.loc[(class0['x1'] == 0) & (class0['x2'] == 0) &
                                    (class0['y1'] == 0) & (class0['y2'] == 0)].index)

    class1 = class1.drop(class1.loc[(class1['x1'] == 0) & (class1['x2'] == 0) &
                                    (class1['y1'] == 0) & (class1['y2'] == 0)].index)

    class2 = class2.drop(class2.loc[(class2['x1'] == 0) & (class2['x2'] == 0) &
                                    (class2['y1'] == 0) & (class2['y2'] == 0)].index)

    class3 = class3.drop(class3.loc[(class3['x1'] == 0) & (class3['x2'] == 0) &
                                    (class3['y1'] == 0) & (class3['y2'] == 0)].index)

    class4 = class4.drop(class4.loc[(class4['x1'] == 0) & (class4['x2'] == 0) &
                                    (class4['y1'] == 0) & (class4['y2'] == 0)].index)

    print('class0 = ', class0.shape, 'class1 = ', class1.shape,
          'class2 = ', class2.shape, 'class3 = ', class3.shape,
          'class4 = ', class4.shape)
    class0.to_csv('csv/class0.csv', index=False)
    class1.to_csv('csv/class1.csv', index=False)
    class2.to_csv('csv/class2.csv', index=False)
    class3.to_csv('csv/class3.csv', index=False)
    class4.to_csv('csv/class4.csv', index=False)
    print('csvs saved!')


def data_preparation(path):
    class0 = pd.read_csv(path + 'csv/class0.csv')
    class1 = pd.read_csv(path + 'csv/class1.csv')
    class2 = pd.read_csv(path + 'csv/class2.csv')
    class3 = pd.read_csv(path + 'csv/class3.csv')
    class4 = pd.read_csv(path + 'csv/class4.csv')
    train_df = pd.concat([class0.iloc[0: int(class0.shape[0] * 0.7), :],
                          class1.iloc[0: int(class1.shape[0] * 0.7), :],
                          class2.iloc[0: int(class2.shape[0] * 0.7), :],
                          class3.iloc[0: int(class3.shape[0] * 0.7), :],
                          class4.iloc[0: int(class4.shape[0] * 0.7), :],
                         ])

    '''
    val_df = pd.concat([class0.iloc[int(class0.shape[0] * 0.7): int(class0.shape[0] * 0.85), :],
                        class1.iloc[int(class1.shape[0] * 0.7): int(class1.shape[0] * 0.85), :],
                        class2.iloc[int(class2.shape[0] * 0.7): int(class2.shape[0] * 0.85), :],
                        class3.iloc[int(class3.shape[0] * 0.7): int(class3.shape[0] * 0.85), :],
                        class4.iloc[int(class4.shape[0] * 0.7): int(class4.shape[0] * 0.85), :]
                       ])
    '''

    test_df = pd.concat([class0.iloc[int(class0.shape[0] * 0.7):, :],
                         class1.iloc[int(class1.shape[0] * 0.7):, :],
                         class2.iloc[int(class2.shape[0] * 0.7):, :],
                         class3.iloc[int(class3.shape[0] * 0.7):, :],
                         class4.iloc[int(class4.shape[0] * 0.7):, :],
                        ])
    return train_df, test_df


def data_windowing(data, seq):
    len = data.shape[0]
    print(len)
    x, y = [], []
    for i in range(len - (seq - 1)):
        x.append(data.iloc[i: i + seq, 0:-1].values)
        y.append(data.iloc[i, -1])
    x = np.array(x)
    y = np.array(y)
    print('shape x: ', x.shape, ' shape y: ', y.shape)
    return x, y


def data_windowing_checker(data):
    print('data_shape:', data.shape)
    for i in range(data.shape[0]):
        yield data[i, ...]



# if __name__ == '__main__':
#    # data = pd.read_csv('csv/class0.csv')
#    train, test = data_preparation()
#    '''
#    x, y = data_windowing(train, 20)
#    d = data_windowing_checker(x)
#    for i in range(100):
#        print(next(d))
#    '''
#    visualization(train)
#    # clean()
#    # class_divider()

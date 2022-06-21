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
    # data = data.loc[~(data == 0).all(axis=1)]
    data = data[data.x1 !=0]
    # data = data.dropna(axis=0)
    print(data.head())
    data = data.astype('float32')
    print(data.describe().transpose())

    for col in data.columns:
        if col != 'label':
            # m = data[col].mean()
            # s = data[col].std()
            # data[col] = (data[col] - m) / s
            # print('mean = ', data[col].mean(), ' std= ', data[col].std())
            mn = data[col].min()
            mx = data[col].max()
            data[col] = (data[col] - mn) / (mx - mn)
            print('min = ', data[col].min(), ' max= ', data[col].max())
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


def visualization_static(data):
    # data = pd.read_csv('csv/clean.csv')
    print(data.head(100))
    t = np.arange(0, data.shape[0])
    l = 0
    u = 2000
    m = data.shape[0]
    # print(type(data['x2'].iloc[100]))

    plt.scatter(t[l: u], data['x2'].iloc[l: u], marker='+')
    plt.scatter(t[l: u], data['x1'].iloc[l: u], marker='+')
    plt.show()


def class_divider():
    data = pd.read_csv('csv/clean.csv')
    print('min = ', data['label'].min(), 'max = ', data['label'].max())
    class0 = data.loc[data['label'] == 0]
    class1 = data.loc[data['label'] == 1]
    # class2 = data.loc[data['label'] == 2]
    # class3 = data.loc[data['label'] == 2] be aware of class label
    class4 = data.loc[data['label'] == 2]

    '''print('class0 = ', class0.shape, 'class1 = ', class1.shape,
          'class2 = ', class2.shape, 'class3 = ', class3.shape,
          'class4 = ', class4.shape)'''

    # print(class0.loc[(class0['x1'] == 0) & (class0['x2'] == 0) & (class0['y1'] == 0) & (class0['y2'] == 0)].index)

    class0 = class0.drop(class0.loc[(class0['x1'] == 0) & (class0['x2'] == 0) &
                                    (class0['y1'] == 0) & (class0['y2'] == 0)].index)

    class1 = class1.drop(class1.loc[(class1['x1'] == 0) & (class1['x2'] == 0) &
                                    (class1['y1'] == 0) & (class1['y2'] == 0)].index)

    # class2 = class2.drop(class2.loc[(class2['x1'] == 0) & (class2['x2'] == 0) &
    #                                (class2['y1'] == 0) & (class2['y2'] == 0)].index)

    # class3 = class3.drop(class3.loc[(class3['x1'] == 0) & (class3['x2'] == 0) &
    #                                 (class3['y1'] == 0) & (class3['y2'] == 0)].index)

    class4 = class4.drop(class4.loc[(class4['x1'] == 0) & (class4['x2'] == 0) &
                                    (class4['y1'] == 0) & (class4['y2'] == 0)].index)

    # print('class0 = ', class0.shape, 'class1 = ', class1.shape,
    #      'class2 = ', class2.shape, 'class3 = ', class3.shape,
    #      'class4 = ', class4.shape)
    class0.to_csv('csv/class0.csv', index=False)
    class1.to_csv('csv/class1.csv', index=False)
    # class2.to_csv('csv/class2.csv', index=False)
    # class3.to_csv('csv/class3.csv', index=False)
    class4.to_csv('csv/class4.csv', index=False)
    print('csvs saved!')


def data_preparation(path):
    class0 = pd.read_csv(path + 'csv/class0.csv')
    class1 = pd.read_csv(path + 'csv/class1.csv')
    # class2 = pd.read_csv(path + 'csv/class2.csv')
    # class3 = pd.read_csv(path + 'csv/class3.csv')
    class4 = pd.read_csv(path + 'csv/class4.csv')
    data = [class0, class1, class4]
    x, y = data_windowing(data, 150)

    train_x = x[0: int(x.shape[0] * 0.7), ...]
    train_y = y[0: int(x.shape[0] * 0.7), ...]

    test_x = x[int(x.shape[0] * 0.7): int(x.shape[0] * 0.85), ...]
    test_y = y[int(x.shape[0] * 0.7): int(x.shape[0] * 0.85), ...]

    val_x = x[int(x.shape[0] * 0.85):, ...]
    val_y = y[int(x.shape[0] * 0.85):, ...]

    print('windower min max = ', x.min(), x.max())

    '''
    train_df = pd.concat([class0.iloc[0: int(class0.shape[0] * 0.7), :],
                          class1.iloc[0: int(class1.shape[0] * 0.7), :],
                          # class2.iloc[0: int(class2.shape[0] * 0.65), :],
                          class3.iloc[0: int(class3.shape[0] * 0.7), :],
                          class4.iloc[0: int(class4.shape[0] * 0.7), :],
                         ])
                         '''
    '''train_df = pd.concat([class0.iloc[int(class0.shape[0] * 0.3):, :],
                          class1.iloc[int(class1.shape[0] * 0.3):, :],
                          # class2.iloc[0: int(class2.shape[0] * 0.65), :],
                          class3.iloc[int(class3.shape[0] * 0.3):, :],
                          class4.iloc[int(class4.shape[0] * 0.3):, :],
                          ])'''

    '''
    val_df = pd.concat([class0.iloc[int(class0.shape[0] * 0.7): int(class0.shape[0] * 0.85), :],
                        class1.iloc[int(class1.shape[0] * 0.7): int(class1.shape[0] * 0.85), :],
                        class2.iloc[int(class2.shape[0] * 0.7): int(class2.shape[0] * 0.85), :],
                        class3.iloc[int(class3.shape[0] * 0.7): int(class3.shape[0] * 0.85), :],
                        class4.iloc[int(class4.shape[0] * 0.7): int(class4.shape[0] * 0.85), :]
                       ])
    '''

    ''' test_df = pd.concat([class0.iloc[0: int(class0.shape[0] * 0.3), :],
                         class1.iloc[0: int(class1.shape[0] * 0.3), :],
                         # class2.iloc[int(class2.shape[0] * 0.65):, :],
                         class3.iloc[0: int(class3.shape[0] * 0.3), :],
                         class4.iloc[0: int(class4.shape[0] * 0.3), :],
                        ])'''
    return train_x, train_y, test_x, test_y, val_x, val_y


def data_windowing(data, seq):
    temp = np.ones([1, seq, data[0].shape[1]])

    print('temp shape: ', temp.shape)
    flag = 1
    for i, patch in enumerate(data):
        d = windower(patch, seq)
        # d = d[np.newaxis, ...]
        # if flag:
        #     temp[0, ...] = d
        #     flag = 0
        # else:
        temp = np.append(temp, d, axis=0)

    temp = np.delete(temp, 0, 0)
    x, y = window_shuffler_labeler(temp)
    y = np.array(y)
    return x, y


def windower(data, seq):
    flag = 1
    len = data.shape[0]
    # print(len)
    d = np.empty([1, seq, data.shape[1]])
    print(d.shape)
    for i in range(len - (seq - 1)):
        t = data.iloc[i: i + seq, :].values
        t = np.array(t)
        print(t.shape)
        t = t[np.newaxis, ...]
        if flag:
            d[0, ...] = t
            flag = 0
        else:
            d = np.append(d, t, axis=0)

    print(d.shape)
    return d


def window_shuffler_labeler(data):
    y = []
    np.random.shuffle(data)
    for i in range(data.shape[0]):
        y.append(data[i, 0, -1])
        print(data[i, 0, -1])
    #               (data, col_num, axis=col) axis=[data, row, col]
    data = np.delete(data, -1, 2)
    # print(data.shape)
    return data, y


def data_windowing_checker(data):
    print('data_shape:', data.shape)
    for i in range(data.shape[0]):
        yield data[i, ...]


if __name__ == '__main__':
    clean()
    class_divider()

    '''data1 = pd.read_csv('csv/class0.csv')
    data2 = pd.read_csv('csv/class1.csv')
    data3 = pd.read_csv('csv/class2.csv')
    data4 = pd.read_csv('csv/class3.csv')
    data5 = pd.read_csv('csv/class4.csv')

    
    print(data1.mean(), data1.std())

    data = [data1, data2, data3, data4, data5]
    x, y = data_windowing(data, 20)
    print(x.shape)
    print(len(y))'''
    # tr_x, tr_y, te_x, te_y = data_preparation('')
    # print('shape train: ', tr_x.shape, tr_y.shape)
    # print('shape test: ', te_x.shape, te_y.shape)

   # visualization(data4)
   # visualization_static
#   visualization_static(data1)
#   visualization_static(data2)
#   visualization_static(data3)
#   # clean()
#   # class_divider()

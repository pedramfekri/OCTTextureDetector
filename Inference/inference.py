import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import xlrd
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf

import IPython
import IPython.display

plt.rcParams.update({'font.size': 16})
pd.set_option('display.max_columns', None)
# read dataset ---------------------------------
data = pd.read_csv('../data/data_v3.csv')
bf_data = data.copy()

# print(data.head())
print(data.shape)
print(data.columns)
unique_date = data['date'].unique()
# print(unique_date.shape, 'type = ', type(unique_date))

unique_date = pd.DataFrame(unique_date)
# date_time = pd.to_datetime(data.pop('date'), format='%d.%m.%Y %H:%M:%S')
# print(unique_date.head())
# time_second = date_time.map(pd.Timestamp.timestamp)
# for d in unique_date:
#     print(d)

# correcting date feature --------------------------------------------------------
new_date = []
for i in range(np.shape(data)[0]):
    d = data.iloc[i]['date']
    ts = pd.DatetimeIndex([d])
    new_date.append(ts)

data['date2'] = pd.DataFrame(new_date)

data = data.sort_values('date2')
data.drop('date', axis=1, inplace=True)

# print("from here:")
# print(data['date2'].head())

date_time = pd.to_datetime(data['date2'], format='%Y-%m-%d %H:%M:%S')
day_year = pd.to_datetime(data['date2'], format='%Y-%m-%d %H:%M:%S').dt.dayofyear

timestamp_s = date_time.map(pd.Timestamp.timestamp)

data['doy'] = pd.DataFrame(day_year)
data['doy2'] = pd.DataFrame(day_year)
data['timestamp'] = pd.DataFrame(timestamp_s)

# data normalization -------------------------------------------------------------------------------------
mean_doy = data['doy'].mean()
std_doy = data['doy'].std()
doy_min = data['doy'].min()
doy_max = data['doy'].max()
data['doy'] = (data['doy'] - doy_min) / (doy_max - doy_min)

print(data['doy'].min())
print(data['doy'].max())

data.drop(['month_of_year', 'week_of_year', 'day_of_week', 'launch_month_net_revenue',
           'date2', 'timestamp', 'doy2', 'nth_discount_since_launch'],
          axis=1, inplace=True)

unique_game = data['sales_title'].unique()
print(unique_game)
le = preprocessing.LabelEncoder()
le.fit(unique_game)
data['sales_title'] = le.transform(data['sales_title'])

unique_game = data['genre'].unique()
print(unique_game)
le = preprocessing.LabelEncoder()
le.fit(unique_game)
data['genre'] = le.transform(data['genre'])

unique_game = data['multiplayer_or_single'].unique()
print(unique_game)
le3 = preprocessing.LabelEncoder()
le3.fit(unique_game)
data['multiplayer_or_single'] = le3.transform(data['multiplayer_or_single'])

unique_game = data['openworld_or_not'].unique()
print(unique_game)
le = preprocessing.LabelEncoder()
le.fit(unique_game)
data['openworld_or_not'] = le.transform(data['openworld_or_not'])

unique_game = data['business_model'].unique()
print(unique_game)
le = preprocessing.LabelEncoder()
le.fit(unique_game)
data['business_model'] = le.transform(data['business_model'])

# data.to_csv('data2.csv', index=False)
print(data.describe().transpose())

# plt.plot(np.array(data['doy']))
# plt.show()

# sns.pairplot(data[['daily_net_revenue', 'days_from_launch', 'nth_day_in_discount_period']], diag_kind='kde')
# plt.show()

mean_sales = data['sales_title'].mean()
std_sales = data['sales_title'].std()
# data['sales_title'] = (data['sales_title'] - mean_sales) / std_sales
sales_min = data['sales_title'].min()
sales_max = data['sales_title'].max()
data['sales_title'] = (data['sales_title'] - sales_min) / (sales_max - sales_min)

print('sales: ', data['sales_title'].min(), ' and ', data['sales_title'].max())

mean_days_from_launch = data['days_from_launch'].mean()
std_days_from_launch = data['days_from_launch'].std()
days_from_launch_min = data['days_from_launch'].min()
days_from_launch_max = data['days_from_launch'].max()
data['days_from_launch'] = (data['days_from_launch']
                            - days_from_launch_min) \
                           / (days_from_launch_max - days_from_launch_min)

mean_number_of_discount_started = data['number_of_discount_started'].mean()
std_number_of_discount_started = data['number_of_discount_started'].std()
number_of_discount_started_min = data['number_of_discount_started'].min()
number_of_discount_started_max = data['number_of_discount_started'].max()
data['number_of_discount_started'] = (data['number_of_discount_started']
                                      - number_of_discount_started_min) \
                                     / (number_of_discount_started_max - number_of_discount_started_min)

mean_genre = data['genre'].mean()
std_genre = data['genre'].std()
genre_min = data['genre'].min()
genre_max = data['genre'].max()
data['genre'] = (data['genre'] - genre_min) / (genre_max - genre_min)

daily_net_revenue_min = data['daily_net_revenue'].min()
daily_net_revenue_max = data['daily_net_revenue'].max()
data['daily_net_revenue'] = (data['daily_net_revenue'] - daily_net_revenue_min) / \
                            (daily_net_revenue_max - daily_net_revenue_min)

mean_nth_day_in_discount_period = data['nth_day_in_discount_period'].mean()
std_nth_day_in_discount_period = data['nth_day_in_discount_period'].std()
nth_day_in_discount_period_min = data['nth_day_in_discount_period'].min()
nth_day_in_discount_period_max = data['nth_day_in_discount_period'].max()
data['nth_day_in_discount_period'] = (data['nth_day_in_discount_period'] -
                                      nth_day_in_discount_period_min) / \
                                     (nth_day_in_discount_period_max - nth_day_in_discount_period_min)

mean_discount_duration = data['discount_duration'].mean()
std_discount_duration = data['discount_duration'].std()
discount_duration_min = data['discount_duration'].min()
discount_duration_max = data['discount_duration'].max()
data['discount_duration'] = (data['discount_duration'] -
                             discount_duration_min) / (discount_duration_max - discount_duration_min)

unique_game_val = data['sales_title'].unique()
game_1 = data.loc[data['sales_title'] == unique_game_val[0]]
game_2 = data.loc[data['sales_title'] == unique_game_val[1]]
game_3 = data.loc[data['sales_title'] == unique_game_val[2]]
game_4 = data.loc[data['sales_title'] == unique_game_val[3]]
game_5 = data.loc[data['sales_title'] == unique_game_val[4]]

unique_game_val2 = bf_data['sales_title'].unique()
bf_game_1 = bf_data.loc[bf_data['sales_title'] == unique_game_val2[0]]
bf_game_2 = bf_data.loc[bf_data['sales_title'] == unique_game_val2[1]]
bf_game_3 = bf_data.loc[bf_data['sales_title'] == unique_game_val2[2]]
bf_game_4 = bf_data.loc[bf_data['sales_title'] == unique_game_val2[3]]
bf_game_5 = bf_data.loc[bf_data['sales_title'] == unique_game_val2[4]]

dropped_feat = ['discounted', 'days_from_launch', 'number_of_discount_started',
                'nth_day_in_discount_period', 'discount_duration',
                'genre', 'multiplayer_or_single', 'openworld_or_not', 'business_model', 'daily_discount_pct',
                'sales_title']

game_1.drop(dropped_feat, axis=1, inplace=True)
game_2.drop(dropped_feat, axis=1, inplace=True)
game_3.drop(dropped_feat, axis=1, inplace=True)
game_4.drop(dropped_feat, axis=1, inplace=True)
game_5.drop(dropped_feat, axis=1, inplace=True)
'''
train_df = game_1.iloc[0: int(game_1.shape[0] * 0.7), :]
val_df = game_1.iloc[int(game_1.shape[0] * 0.7): int(game_1.shape[0] * 0.85), :]
test_df = game_1.iloc[int(game_1.shape[0] * 0.85):, :]
'''
train_df = pd.concat([
    game_1.iloc[0: int(game_1.shape[0] * 0.7), :],
    game_2.iloc[0: int(game_2.shape[0] * 0.7), :],
    game_3.iloc[0: int(game_3.shape[0] * 0.7), :],
    game_4.iloc[0: int(game_4.shape[0] * 0.7), :],
    game_5.iloc[0: int(game_5.shape[0] * 0.7), :],
])

bf_train = pd.concat([
    bf_game_1.iloc[0: int(bf_game_1.shape[0] * 0.7), :],
    bf_game_2.iloc[0: int(bf_game_2.shape[0] * 0.7), :],
    bf_game_3.iloc[0: int(bf_game_3.shape[0] * 0.7), :],
    bf_game_4.iloc[0: int(bf_game_4.shape[0] * 0.7), :],
    bf_game_5.iloc[0: int(bf_game_5.shape[0] * 0.7), :],
])

val_df = pd.concat([
    game_1.iloc[int(game_1.shape[0] * 0.7): int(game_1.shape[0] * 0.85), :],
    game_2.iloc[int(game_2.shape[0] * 0.7): int(game_2.shape[0] * 0.85), :],
    game_3.iloc[int(game_3.shape[0] * 0.7): int(game_3.shape[0] * 0.85), :],
    game_4.iloc[int(game_4.shape[0] * 0.7): int(game_4.shape[0] * 0.85), :],
    game_5.iloc[int(game_5.shape[0] * 0.7): int(game_5.shape[0] * 0.85), :]
])

bf_val_df = pd.concat([
    bf_game_1.iloc[int(bf_game_1.shape[0] * 0.7): int(bf_game_1.shape[0] * 0.85), :],
    bf_game_2.iloc[int(bf_game_2.shape[0] * 0.7): int(bf_game_2.shape[0] * 0.85), :],
    bf_game_3.iloc[int(bf_game_3.shape[0] * 0.7): int(bf_game_3.shape[0] * 0.85), :],
    bf_game_4.iloc[int(bf_game_4.shape[0] * 0.7): int(bf_game_4.shape[0] * 0.85), :],
    bf_game_5.iloc[int(bf_game_5.shape[0] * 0.7): int(bf_game_5.shape[0] * 0.85), :]
])

test_df = pd.concat([
    game_1.iloc[int(game_1.shape[0] * 0.85):, :],
    game_2.iloc[int(game_2.shape[0] * 0.85):, :],
    game_3.iloc[int(game_3.shape[0] * 0.85):, :],
    game_4.iloc[int(game_4.shape[0] * 0.85):, :],
    game_5.iloc[int(game_5.shape[0] * 0.85):, :],
])

bf_test_df = pd.concat([
    bf_game_1.iloc[int(bf_game_1.shape[0] * 0.85):, :],
    bf_game_2.iloc[int(bf_game_2.shape[0] * 0.85):, :],
    bf_game_3.iloc[int(bf_game_3.shape[0] * 0.85):, :],
    bf_game_4.iloc[int(bf_game_4.shape[0] * 0.85):, :],
    bf_game_5.iloc[int(bf_game_5.shape[0] * 0.85):, :],
])

# sns.pairplot(data[['daily_net_revenue', 'days_from_launch', 'nth_day_in_discount_period']], diag_kind='kde')
# plt.show()
# data.drop([ 'doy'], axis=1, inplace=True)
# data.to_csv('data2.csv', index=False)
print(data.describe().transpose())

# print("game_1 ", game_1.shape, "game_2 ", game_2.shape, "game_3 ", game_3.shape, "game_4 ", game_4.shape)

# split dataset
df = train_df
column_indices = {name: i for i, name in enumerate(df.columns)}
num_features = df.shape[1]


# window generating
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns='daily_net_revenue'):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


w1 = WindowGenerator(input_width=15, label_width=1, shift=1,
                     label_columns=['daily_net_revenue'])
w1


# split window
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100 + w1.total_window_size]),
                           np.array(train_df[200:200 + w1.total_window_size])])

example_inputs, example_labels = w1.split_window(example_window)

print('All shapes are: (batch, day, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# plotting
w1.example = example_inputs, example_labels


def plot(self, model=None, plot_col='daily_net_revenue', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col}')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [day]')


WindowGenerator.plot = plot


# creating tf dataset
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=1, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair.
w1.train.element_spec

# iterating over dataset
for example_inputs, example_labels in w1.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

OUT_STEPS = 3
multi_window = WindowGenerator(input_width=15,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=["daily_net_revenue"])

# multi_window.plot()
multi_window

# RNN

model = tf.keras.models.load_model('../savedModel/')
# print(next(iter(multi_window.train)))

dd = iter(multi_window.test)
# feat, lab = next(dd)
# print('input shape: ', feat.shape, '  label shape: ', lab.shape)
# print('label_normal ', lab)
MIN = 0
MAX = 68884.21063
all_pred = []

label_normal = np.array([[0, 0, 0]])
label_denormal = np.array([[0, 0, 0]])
pred_normal = np.array([[0, 0, 0]])
pred_denormal = np.array([[0, 0, 0]])
label_denormal = label_denormal.transpose()
label_normal = label_normal.transpose()
pred_normal = pred_normal.transpose()
pred_denormal = pred_denormal.transpose()
feat, lab = next(dd)
print(feat.numpy().shape)
print(lab.numpy().shape)

for i in range(len(test_df)-20):
    # i += 21
    feat, lab = next(dd)
    # if(i > 0 and i < 1000):
    pred = model.predict(feat)
    pred = pred[np.newaxis, ...]
    lab = lab[np.newaxis, ...]
    pred = pred[0, 0, :, :]
    lab = lab[0, 0, :, :]
    print(pred.shape)
    print(pred_normal.shape)
    # pred_normal = np.append(pred_normal, pred)
    pred_normal = np.concatenate((pred_normal, pred), axis=1)

    # label_normal = np.append(label_normal, lab)
    label_normal = np.concatenate((label_normal, lab), axis=1)
    print('------label index:', i)
    # print("net prediction: ", pred, "  true label:  ", lab)
    # pred = pred.numpy()

    pred_net_estimate = pred * (MAX - MIN) + MIN
    lab_denorm = lab * (MAX - MIN) + MIN
    # pred_denormal = np.append(pred_denormal, pred_net_estimate)
    # label_denormal = np.append(label_denormal, lab_denorm)
    pred_denormal = np.concatenate((pred_denormal, pred_net_estimate), axis=1)
    label_denormal = np.concatenate((label_denormal, lab_denorm), axis=1)
    # print("denormal net prediction: ", pred_net_estimate)
    print(feat[:, :, 1].shape)
    # plt.plot(np.arange(feat.numpy().shape[1]), np.squeeze(feat[:, :, 1]))
    a = feat.numpy().shape[1]
    b = feat.numpy().shape[1] + lab.numpy().shape[1]
    print(a, b)

print("shape pred_no", pred_normal.shape)
print("shape pred_deno", pred_denormal.shape)
print("shape lab_no", label_normal.shape)
print("shape lab_deno", label_denormal.shape)

'''
plt.subplot(2, 1, 1)
plt.plot(pred_normal, label='pred')
plt.plot(label_normal, label='label')
plt.subplot(2, 1, 2)
plt.plot(pred_denormal, label='pred')
plt.plot(label_denormal, label='label')
plt.show()'''

print(pred_normal.shape)
fig, ax = plt.subplots(3, 1)
# plt.subplot(3, 1, 1)
ax[0].plot(pred_normal[0, :], label='pred_step1')
ax[0].plot(label_normal[0, :], label='label')
# plt.subplot(3, 1, 2)
ax[1].plot(pred_normal[1, :], label='pred_step2')
ax[1].plot(label_normal[1, :], label='label')
# plt.subplot(3, 1, 3)
ax[2].plot(pred_normal[2, :], label='pred_step3')
ax[2].plot(label_normal[2, :], label='label')
ax[0].set_title("one step ahead")
ax[1].set_title("two step ahead")
ax[2].set_title("three step ahead")

fig1, ax1 = plt.subplots(3, 1)
# plt.subplot(3, 1, 1)
ax1[0].plot(pred_denormal[0, :], label='pred_step1')
ax1[0].plot(label_denormal[0, :], label='label')
# plt.subplot(3, 1, 2)
ax1[1].plot(pred_denormal[1, :], label='pred_step2')
ax1[1].plot(label_denormal[1, :], label='label')
# plt.subplot(3, 1, 3)
ax1[2].plot(pred_denormal[2, :], label='pred_step3')
ax1[2].plot(label_denormal[2, :], label='label')
ax1[0].set_title("one step ahead")
ax1[1].set_title("two step ahead")
ax1[2].set_title("three step ahead")
plt.show()

pred_normal = pred_normal.transpose()
pred_normal = pd.DataFrame(pred_normal)
pred_normal = pred_normal.rename(columns={0:'pred_normal_0', 1:'pred_normal_1', 2:'pred_normal_2'})

label_normal = label_normal.transpose()
label_normal = pd.DataFrame(label_normal)
label_normal = label_normal.rename(columns={0:'label_normal_0', 1:'label_normal_1', 2:'label_normal_2'})

pred_normal = pred_normal.join(label_normal)
pred_normal.to_csv('../preds/normal_3stepahead2.csv', index=False)

pred_denormal = pred_denormal.transpose()
pred_denormal = pd.DataFrame(pred_denormal)
pred_denormal = pred_denormal.rename(columns={0:'pred_normal_0', 1:'pred_normal_1', 2:'pred_normal_2'})

label_denormal = label_denormal.transpose()
label_denormal = pd.DataFrame(label_denormal)
label_denormal = label_denormal.rename(columns={0:'label_normal_0', 1:'label_normal_1', 2:'label_normal_2'})

pred_denormal = pred_denormal.join(label_denormal)
pred_denormal.to_csv('../preds/denormal_3stepahead2.csv', index=False)

dn = np.array([pred_denormal, label_denormal])
n = np.array([pred_normal, label_normal])
# np.savetxt("denorm_pred_label_all_games.csv", dn, delimiter=",")
# np.savetxt("norm_pred_label_all_games.csv", n, delimiter=",")

# denormalization


# print(model.summary())

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
data = pd.read_csv('../data/data_v3.csv')
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
# data.to_csv('data2.csv')

# interval = np.pi * 2
# t = interval / 365
# data['doy'] = np.sin(data['doy'] * t)

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
'''
plt.plot(np.array(data['doy']))
plt.show()

sns.pairplot(data[['daily_net_revenue', 'days_from_launch', 'nth_day_in_discount_period']], diag_kind='kde')
plt.show()
'''
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

dropped_feat = ['discounted', 'days_from_launch', 'number_of_discount_started',
                'nth_day_in_discount_period', 'discount_duration',
                'genre', 'multiplayer_or_single', 'openworld_or_not', 'business_model', 'daily_discount_pct', 'sales_title']

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

val_df = pd.concat([
                    game_1.iloc[int(game_1.shape[0] * 0.7): int(game_1.shape[0] * 0.85), :],
                    game_2.iloc[int(game_2.shape[0] * 0.7): int(game_2.shape[0] * 0.85), :],
                    game_3.iloc[int(game_3.shape[0] * 0.7): int(game_3.shape[0] * 0.85), :],
                    game_4.iloc[int(game_4.shape[0] * 0.7): int(game_4.shape[0] * 0.85), :],
                    game_5.iloc[int(game_5.shape[0] * 0.7): int(game_5.shape[0] * 0.85), :]
                    ])

test_df = pd.concat([
                     game_1.iloc[int(game_1.shape[0] * 0.85):, :],
                     game_2.iloc[int(game_2.shape[0] * 0.85):, :],
                     game_3.iloc[int(game_3.shape[0] * 0.85):, :],
                     game_4.iloc[int(game_4.shape[0] * 0.85):, :],
                     game_5.iloc[int(game_5.shape[0] * 0.85):, :],
                     ])


# sns.pairplot(data[['daily_net_revenue', 'days_from_launch', 'nth_day_in_discount_period']], diag_kind='kde')
# plt.show()
# data.drop([ 'doy'], axis=1, inplace=True)
# data.to_csv('data2.csv', index=False)
print(data.describe().transpose())

# print("game_1 ", game_1.shape, "game_2 ", game_2.shape, "game_3 ", game_3.shape, "game_4 ", game_4.shape)

'''plt.plot(np.array(game_1['doy']))
plt.show()
plt.plot(np.array(game_2['doy']))
plt.show()
plt.plot(np.array(game_3['doy']))
plt.show()
plt.plot(np.array(game_4['doy']))
plt.show()'''

# print("game_1 ", game_1['doy'].values[0], "  ", game_1['doy'].values[-1],
#       "\ngame_2 ", game_2['doy'].values[0], "  ", game_2['doy'].values[-1],
#       "\ngame_3 ", game_3['doy'].values[0], "  ", game_3['doy'].values[-1],
#       "\ngame_4 ", game_4['doy'].values[0], "  ", game_4['doy'].values[-1])

# dataset = pd.concat([game_1, game_2, game_3, game_4])
# dataset = game_1
# dataset = pd.DataFrame(dataset['daily_net_revenue'])
# print(dataset.columns)
# plt.plot(np.array(dataset['doy']))
# plt.show()


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

# w1.plot()


# creating tf dataset
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=32, )

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

print(multi_window.train)


MAX_EPOCHS = 200


def compile_and_fit(model, window, patience=2):
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                   patience=patience,
    #                                                   mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        # callbacks=[early_stopping]
    )
    model.save('../savedModel')
    return history, model


# IPython.display.clear_output()

# RNN
multi_lstm_model = tf.keras.Sequential([

    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    # tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    # tf.keras.layers.Dropout(0.01),
    # Shape => [batch, out_steps*features].
    # tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(OUT_STEPS * 1, activation='relu'),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, 1])
])

history, model = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_window.plot(multi_lstm_model)

'''
# plotting performance in history
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
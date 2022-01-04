import tensorflow as tf
from Dataset import dataset as data
from Model import model

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


if __name__ == '__main__':
    path = '../Dataset/'
    time_step = 20
    train, test = data.data_preparation(path)
    x_train, y_train = data.data_windowing(train, time_step)
    x_test, y_test = data.data_windowing(test, time_step)

    model = model.lstm_classification([32], [16], time_step, 4, 5)
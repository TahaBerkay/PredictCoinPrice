from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D

params = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.00010000,
    "window_size": 30,  # cnn_n_seq * cnn_n_steps = window_size
    "cnn_n_seq": 6,
    "cnn_n_steps": 5,
    "n_features": 5
}

input_format = (params["window_size"], params["n_features"])
prediction_input_format = (1, params["window_size"], params["n_features"])
cnn_input_format = (params["cnn_n_seq"], params["cnn_n_steps"], params["n_features"])
prediction_cnn_input_format = (1, params["cnn_n_seq"], params["cnn_n_steps"], params["n_features"])
conv_lstm_input_format = (params["cnn_n_seq"], 1, params["cnn_n_steps"], params["n_features"])
prediction_conv_lstm_input_format = (1, params["cnn_n_seq"], 1, params["cnn_n_steps"], params["n_features"])


def vanilla_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_format))
    model.add(Dense(1))  # eğer 2 adım istersen next 2 mins gibi
    model.compile(optimizer='adam', loss='mse')
    return model


def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_format))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def bidirectional_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=input_format))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def cnn_lstm():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=cnn_input_format))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def conv_lstm():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=conv_lstm_input_format))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def advanced_cnn_lstm():
    x = []


'''
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=5,
                           strides=1,
                           padding="causal",
                           activation="relu",
                           input_shape=windowed_dataset_train.shape[-2:]),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
'''


def advanced_lstm():
    x = []


'''
    lstm_model = Sequential()
    # model.add(LSTM(input_shape=(None, input_dim), units=output_dim))
    lstm_model.add(LSTM(128, input_shape=input_format))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(LSTM(64, input_shape=(1, params["window_size"]), dropout=0.0))
    # lstm_model.add(Dropout(0.4))
    # lstm_model.add(Dense(20, activation='relu'))
    # lstm_model.add(Dense(1, activation='sigmoid'))
    # model.add(Activation('linear'))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    return lstm_model
'''

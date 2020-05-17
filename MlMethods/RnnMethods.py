import os
import pickle

import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import CustomSettings
from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods, RnnModelCreators


class LstmMethod(Methods.Method):
    window_size = RnnModelCreators.params["window_size"]
    train_cols = ["rsi", "roc", "ema", "macd", "so"]

    def __init__(self, data, data_interval):
        super().__init__(data, data_interval)
        self.min_max_scaler = None
        self.input_windows = []
        self.next_price = None
        self.model_data_path = os.path.join(CustomSettings.DATAFILES_DIR,
                                            self.__class__.__name__ + '_scaler_' + self.data_interval.name + '.pkl')

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def forecast(self, nb_of_steps):
        return self.make_prediction(RnnModelCreators.prediction_input_format).item(0)

    def make_prediction(self, input_format):
        input = DatasetProcessor.preprocess_input_data(self.data)[self.train_cols][-self.window_size:]
        transformed_input = self.min_max_scaler["in"].transform(input)
        reshaped_input = transformed_input.reshape(input_format)
        # y_pred = self.model.predict(reshaped_input)
        # y_pred = y_pred.flatten().reshape(-1, 1)
        # return pd.np.argmax(y_pred)
        return self.model.predict_classes(reshaped_input)

    def prepare_data_before_train(self):
        self.scale_data()
        self.prepare_windowed_input()

    def scale_data(self):
        data = self.data[: -1].values

        self.next_price = DatasetProcessor.prepare_labels(self.data)[:-self.window_size]
        min_max_scaler_out = LabelEncoder().fit(self.next_price)
        self.next_price = min_max_scaler_out.transform(self.next_price)

        self.data = DatasetProcessor.preprocess_input_data(self.data)[self.train_cols][:-1]
        min_max_scaler_in = MinMaxScaler().fit(self.data)
        self.data = min_max_scaler_in.transform(self.data)

        self.min_max_scaler = dict({"in": min_max_scaler_in, "out": min_max_scaler_out})

    def prepare_windowed_input(self):
        last_interval_start_idx = self.data.shape[0] - self.window_size
        for start_idx in range(last_interval_start_idx):
            end_idx = start_idx + self.window_size
            self.input_windows.append(self.data[start_idx:end_idx, :])
        self.input_windows = pd.np.array(self.input_windows)

    def save_model(self):
        super().save_model()
        path = os.path.join(self.model_data_path)
        with open(path, Methods.file_mode_write_binary) as pkl:
            pickle.dump(self.min_max_scaler, pkl)

    def load_model(self):
        super().load_model()
        path = os.path.join(self.model_data_path)
        with open(path, Methods.file_mode_read_binary) as pkl:
            self.min_max_scaler = pickle.load(pkl)


class VanillaLstm(LstmMethod):
    def fit_model(self):
        self.prepare_data_before_train()
        self.model = RnnModelCreators.vanilla_lstm()
        self.model.fit(self.input_windows, self.next_price, epochs=RnnModelCreators.params["epochs"], verbose=2)


class StackedLstm(LstmMethod):
    def fit_model(self):
        self.prepare_data_before_train()
        self.model = RnnModelCreators.stacked_lstm()
        self.model.fit(self.input_windows, self.next_price, epochs=RnnModelCreators.params["epochs"], verbose=2)


class BidirectionalLstm(LstmMethod):
    def fit_model(self):
        self.prepare_data_before_train()
        self.model = RnnModelCreators.bidirectional_lstm()
        self.model.fit(self.input_windows, self.next_price, epochs=RnnModelCreators.params["epochs"], verbose=2)


class CnnLstm(LstmMethod):
    def fit_model(self):
        self.prepare_data_before_train()
        self.input_windows = self.input_windows.reshape((self.input_windows.shape[0],) +
                                                        RnnModelCreators.cnn_input_format)
        self.model = RnnModelCreators.cnn_lstm()
        self.model.fit(self.input_windows, self.next_price, epochs=RnnModelCreators.params["epochs"], verbose=2)

    def forecast(self, nb_of_steps):
        return self.make_prediction(RnnModelCreators.prediction_cnn_input_format).item(0)


class ConvLstm(LstmMethod):
    def fit_model(self):
        self.prepare_data_before_train()
        self.input_windows = self.input_windows.reshape((self.input_windows.shape[0],) +
                                                        RnnModelCreators.conv_lstm_input_format)
        self.model = RnnModelCreators.conv_lstm()
        self.model.fit(self.input_windows, to_categorical(self.next_price), epochs=RnnModelCreators.params["epochs"],
                       verbose=2)

    def forecast(self, nb_of_steps):
        return self.make_prediction(RnnModelCreators.prediction_conv_lstm_input_format).item(0)

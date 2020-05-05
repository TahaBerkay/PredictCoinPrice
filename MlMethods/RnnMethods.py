from pandas import np
from sklearn.preprocessing import MinMaxScaler

from Enums import Decision
from MlMethods import Methods

import pandas as pd


class LstmMethod(Methods.Method):
    window_size = 30

    def manipulate_data(self):
        #data_labels = self.data['Close'].shift(-1)[:-1]
        self.data = self.data.drop("Date", axis=1, inplace=False)
        self.data = self.data[: -1]
        #x = self.data.values
        #min_max_scaler = MinMaxScaler()
        # min_max_scaler.fit()
        # min_max_scaler.transform()
        # min_max_scaler.inverse_transform()
        #self.data = min_max_scaler.fit_transform(x)
        next_price = self.data[self.window_size:]['Close']
        input_windows = []
        last_interval_start_idx = self.data.shape[0] - self.window_size
        for start_idx in range(last_interval_start_idx):
            end_idx = start_idx + self.window_size
            input_windows.append(self.data.iloc[start_idx:end_idx, :])
        self.data = pd.DataFrame({'input_windows': input_windows, 'next_price': next_price})


    def fit_model(self):
        data_labels = np.sign(np.diff(self.data['Close'].to_numpy())) + 1


    def forecast(self, nb_of_steps):
        predictions = self.model.predict(self.data.iloc[-1].to_numpy().reshape(1, -1))
        return Decision(predictions).name


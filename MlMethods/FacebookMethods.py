import pandas as pd
from fbprophet import Prophet

import CustomSettings
from Enums import DataInterval
from MlMethods import Methods

pandas_date_range_mapping = {
    DataInterval.MIN_ONE: 'min',
    DataInterval.MIN_FIVE: '5min',
    DataInterval.MIN_FIFTEEN: '15min',
    DataInterval.MIN_THIRTY: '30min',
    DataInterval.HOUR_ONE: 'H',
    DataInterval.HOUR_TWO: '2H',
    DataInterval.HOUR_FOUR: '4H',
    DataInterval.HOUR_SIX: '6H',
    DataInterval.HOUR_EIGHT: '8H',
    DataInterval.HOUR_TWELVE: '12H',
    DataInterval.DAY_ONE: 'D',
    DataInterval.DAY_THREE: '3D',
    DataInterval.WEEK_ONE: 'W',
    DataInterval.MONTH_ONE: 'M',
}


class ProphetMethod(Methods.Method):
    window_size = CustomSettings.WINDOW_SIZE

    def manipulate_data(self):
        train_dataset = pd.DataFrame()
        train_dataset['ds'] = self.data["Date"]
        train_dataset['y'] = self.data["Close"]
        start_idx = train_dataset.shape[0] % self.window_size
        self.data = train_dataset[start_idx:]

    def fit_model(self):
        self.model = Prophet()
        nb_of_windows = int(self.data.shape[0] / self.window_size)
        self.model.fit(self.data.iloc[0:self.window_size])
        for window_idx in range(1, nb_of_windows):
            start_idx = window_idx * self.window_size
            self.model.fit(self.data.iloc[start_idx:start_idx + self.window_size], init=self.stan_init())

    def forecast(self, nb_of_steps):
        stan_init = self.stan_init()
        self.model = Prophet()
        self.model.fit(self.data, init=stan_init)
        df_future = self.model.make_future_dataframe(periods=nb_of_steps,
                                                     freq=pandas_date_range_mapping[self.data_interval],
                                                     include_history=False)
        df_forecast = self.model.predict(df_future)
        return df_forecast['yhat'].values

    def stan_init(self):
        result = {}
        for param_name in ['k', 'm', 'sigma_obs']:
            result[param_name] = self.model.params[param_name][0][0]
        for param_name in ['delta', 'beta']:
            result[param_name] = self.model.params[param_name][0]
        return result

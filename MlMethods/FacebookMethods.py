import pandas as pd
from fbprophet import Prophet

from MlMethods import Methods


class ProphetMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def manipulate_data(self):
        train_dataset = pd.DataFrame()
        train_dataset['ds'] = self.data["Date"]
        train_dataset['y'] = self.data["Close"]
        self.data = train_dataset

    def fit_model(self):
        self.model = Prophet()
        self.model.fit(self.data)
        # m.add_seasonality(name='monthly', period=21)

    def forecast(self, nb_of_steps):
        stan_init = self.stan_init()
        self.model = Prophet()
        self.model.fit(self.data, init=stan_init)
        df_future = self.model.make_future_dataframe(periods=nb_of_steps, freq='min', include_history=False)
        df_forecast = self.model.predict(df_future)
        return df_forecast['yhat'].values

    def stan_init(self):
        result = {}
        for param_name in ['k', 'm', 'sigma_obs']:
            result[param_name] = self.model.params[param_name][0][0]
        for param_name in ['delta', 'beta']:
            result[param_name] = self.model.params[param_name][0]
        return result

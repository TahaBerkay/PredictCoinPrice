import pandas as pd
from fbprophet import Prophet

from MlMethods import Methods


class ProphetMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        train_dataset = pd.DataFrame()
        train_dataset['ds'] = self.data["Date"]
        train_dataset['y'] = self.data["Close"]
        self.model = Prophet()
        self.model.fit(train_dataset)
        # m.add_seasonality(name='monthly', period=21)

    def forecast(self, nb_of_steps):
        df_future = self.model.make_future_dataframe(periods=nb_of_steps, freq='min', include_history=False)
        df_forecast = self.model.predict(df_future)
        return df_forecast['yhat']

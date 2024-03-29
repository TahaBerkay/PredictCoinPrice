from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class ArimaMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data['Close']

    def fit_model(self):
        model_arima = ARIMA(self.data, order=(2, 1, 0))
        self.model = model_arima.fit(disp=-1)

    def forecast(self):
        predictions = self.model.forecast(steps=1)
        return predictions[0]


class AutoArimaMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data['Close']

    def fit_model(self):
        model_auto_arima = auto_arima(self.data,
                                      start_p=0, start_q=0,
                                      test='adf',
                                      max_p=3, max_q=3,
                                      m=1,
                                      d=None,
                                      seasonal=False,
                                      start_P=0,
                                      D=0,
                                      trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)
        self.model = model_auto_arima.fit(y=self.data, disp=-1)

    def forecast(self):
        self.model.update(self.data)
        predictions = self.model.predict(n_periods=1)
        return DatasetProcessor.prepare_result(predictions[0], self.data.tail(1).values[0])

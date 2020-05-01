from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

from MlMethods import Methods


class ArimaMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        modelARIMA = ARIMA(self.data, order=(2, 1, 0))
        self.model = modelARIMA.fit(disp=-1)

    def forecast(self, nbOfSteps):
        predictions = self.model.forecast(steps=nbOfSteps)
        return predictions


class AutoArimaMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        modelAutoARIMA = auto_arima(self.data,
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
        self.model = modelAutoARIMA.fit(y=self.data, disp=-1)

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(n_periods=nbOfSteps)
        return predictions

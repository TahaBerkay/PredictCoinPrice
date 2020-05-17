import pandas as pd

import CustomSettings
from DatasetHandler.TechnicalIndicatorHelper import TechnicalIndicatorHelper

long_periods = CustomSettings.WINDOW_SIZE
short_periods = CustomSettings.SHORT_PERIODS
signal_periods = CustomSettings.SIGNAL_PERIODS


class DatasetProcessor:

    @staticmethod
    def preprocess_input_data(data):
        input = TechnicalIndicatorHelper.prepare_indicator_based_input_data(data)
        return input

    @staticmethod
    def prepare_labels(data):
        label = (pd.np.sign(pd.np.diff(data['Close'].to_numpy())) + 1)[long_periods + short_periods:]
        # label = data["Close"].shift(-1)[:-1].iloc[long_periods+short_periods:].reset_index()
        return label.astype(int)

    @staticmethod
    def prepare_result(prediction, previous):
        return (pd.np.sign(prediction - previous) + 1).astype(int)

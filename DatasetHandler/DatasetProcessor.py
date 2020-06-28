import pandas as pd

import CustomSettings
from DatasetHandler.TechnicalIndicatorHelper import TechnicalIndicatorHelper

long_periods = CustomSettings.WINDOW_SIZE
short_periods = CustomSettings.SHORT_PERIODS
signal_periods = CustomSettings.SIGNAL_PERIODS


class DatasetProcessor:
    self_hold_range = CustomSettings.HOLD_RANGE

    @staticmethod
    def preprocess_input_data(data):
        input = TechnicalIndicatorHelper.prepare_important_indicators_based_input_data(data)
        return input

    @staticmethod
    def prepare_labels(data):
        hold_ranges = (data['Close'] * DatasetProcessor.self_hold_range)[:-1]
        close_diffs = pd.np.diff(data['Close'].to_numpy())
        label_list = [int(pd.np.sign(close_diff) + 1 if abs(close_diff) > hold_range else 1) for close_diff, hold_range
                      in zip(close_diffs, hold_ranges)]
        label = label_list[long_periods + short_periods:]
        # label = data["Close"].shift(-1)[:-1].iloc[long_periods+short_periods:].reset_index()
        return label

    @staticmethod
    def prepare_result(prediction, previous):
        diff = prediction - previous
        label = pd.np.sign(diff) + 1 if abs(diff) > (prediction * DatasetProcessor.self_hold_range) else 1
        return label

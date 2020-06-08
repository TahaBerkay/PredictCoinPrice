import pandas as pd
import ta

import CustomSettings

long_periods = CustomSettings.WINDOW_SIZE
short_periods = CustomSettings.SHORT_PERIODS
signal_periods = CustomSettings.SIGNAL_PERIODS


class TechnicalIndicatorHelper:

    @staticmethod
    def prepare_important_indicators_based_input_data(data):
        adxi = TechnicalIndicatorHelper.average_directional_movement_index(data)
        cci = TechnicalIndicatorHelper.commodity_channel_index(data)
        rsi = TechnicalIndicatorHelper.relative_strength_index(data)
        macd = TechnicalIndicatorHelper.moving_average_convergence_divergence(data)
        so = TechnicalIndicatorHelper.stochastic_oscillator(data)
        return \
            pd.concat([adxi.rename('adxi'), cci.rename('cci'), rsi, macd.rename('macd'), so.rename('so')], axis=1).iloc[
            long_periods + short_periods:].replace([pd.np.inf, -pd.np.inf], pd.np.nan).fillna(0)[
                ['adxi', 'cci', 'rsi', 'macd', 'so']]

    @staticmethod
    def prepare_indicator_based_input_data(data):
        rsi = TechnicalIndicatorHelper.relative_strength_index(data)
        roc = TechnicalIndicatorHelper.rate_of_change(data)
        ema = TechnicalIndicatorHelper.exponential_moving_average(data)
        macd = TechnicalIndicatorHelper.moving_average_convergence_divergence(data)
        so = TechnicalIndicatorHelper.stochastic_oscillator(data)
        return pd.concat([rsi, roc, ema.rename('ema'), macd.rename('macd'), so.rename('so')], axis=1).iloc[
               long_periods + short_periods:].reset_index().fillna(0)

    @staticmethod
    def prepare_indicator_based_input_data_detailed(data):
        rsi = TechnicalIndicatorHelper.relative_strength_index(data)
        roc = TechnicalIndicatorHelper.rate_of_change(data)
        tsi = TechnicalIndicatorHelper.true_strength_index(data)

        ema = TechnicalIndicatorHelper.exponential_moving_average(data)
        macd = TechnicalIndicatorHelper.moving_average_convergence_divergence(data)
        adxi = TechnicalIndicatorHelper.average_directional_movement_index(data)

        wr = TechnicalIndicatorHelper.williams_r_indicator(data)
        so = TechnicalIndicatorHelper.stochastic_oscillator(data)
        return pd.concat([rsi, roc, tsi, ema, macd, adxi, wr, so], axis=1)

    @staticmethod
    def commodity_channel_index(data):
        return ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'],
                                     n=long_periods).cci()

    @staticmethod
    def rate_of_change(data):
        return ta.momentum.roc(close=data['Close'], n=long_periods)

    @staticmethod
    def relative_strength_index(data):
        return ta.momentum.RSIIndicator(close=data['Close'], n=long_periods).rsi()

    @staticmethod
    def exponential_moving_average(data):
        return ta.trend.EMAIndicator(close=data['Close'], n=long_periods).ema_indicator()

    @staticmethod
    def moving_average_convergence_divergence(data):
        return ta.trend.MACD(close=data['Close'], n_fast=short_periods, n_slow=long_periods,
                             n_sign=signal_periods).macd()

    @staticmethod
    def average_directional_movement_index(data):
        return ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'],
                                     n=long_periods).adx()

    @staticmethod
    def true_strength_index(data):
        return ta.momentum.TSIIndicator(close=data['Close'], r=long_periods, s=short_periods).tsi()

    @staticmethod
    def williams_r_indicator(data):
        return ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'],
                                              lbp=long_periods).wr()

    @staticmethod
    def stochastic_oscillator(data):
        return ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'],
                                                n=long_periods).stoch()

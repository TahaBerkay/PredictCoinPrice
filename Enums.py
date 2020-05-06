import enum


class Decision(enum.Enum):
    SELL = 0
    HOLD = 1
    BUY = 2


class Method(enum.Enum):
    AUTO_ARIMA = 1
    HMM = 2
    LIGHT_GBM = 3
    PROPHET = 4
    SVM = 5
    SVR = 6
    VANILLA_LSTM = 7
    STACKED_LSTM = 8
    BIDIRECTIONAL_LSTM = 9
    CNN_LSTM = 10
    CONV_LSTM = 11


class RunningMode(enum.Enum):
    TRAIN = 1
    PREDICT = 2


class DataInterval(enum.Enum):
    MIN_ONE = 1
    MIN_FIVE = 2
    MIN_FIFTEEN = 3
    MIN_THIRTY = 4
    HOUR_ONE = 5
    HOUR_TWO = 6
    HOUR_FOUR = 7
    HOUR_SIX = 8
    HOUR_EIGHT = 9
    HOUR_TWELVE = 10
    DAY_ONE = 11
    DAY_THREE = 12
    WEEK_ONE = 13
    MONTH_ONE = 14


class CsvColumns(enum.Enum):
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4

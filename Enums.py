import enum


class Decision(enum.Enum):
    BUY = 1
    HOLD = 2
    SELL = 3


class Method(enum.Enum):
    ARIMA = 1
    AUTO_ARIMA = 2
    HMM = 3
    SVM = 6
    SVR = 7
    LIGHT_GBM = 8
    PROPHET = 9


class RunningMode(enum.Enum):
    TRAIN = 1
    PREDICT = 2
    TRAIN_AND_PREDICT = 3


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

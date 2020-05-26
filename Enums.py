import enum


class Decision(enum.Enum):
    SELL = 0
    HOLD = 1
    BUY = 2


class Method(enum.Enum):
    AUTO_ARIMA = 1
    HMM = 2
    LIGHT_GBM = 3
    XGBOOST = 4
    PROPHET = 5
    SVM = 6
    KNN_CLASSIFIER = 7
    RANDOMFOREST_CLASSIFIER = 8
    LOGISTIC_REGRESSOR = 9
    GAUSSIAN_NB = 10
    BERNOULLI_NB = 11
    SVR = 12
    KNN_REGRESSOR = 13
    LINEAR_REGRESSOR = 14
    RANDOMFOREST_REGRESSOR = 15
    VANILLA_LSTM = 20
    STACKED_LSTM = 21
    BIDIRECTIONAL_LSTM = 22
    CNN_LSTM = 23
    CONV_LSTM = 24


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

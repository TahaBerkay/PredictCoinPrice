import enum


class Decision(enum.Enum):
    BUY = 1
    HOLD = 2
    SELL = 3


class Method(enum.Enum):
    ARIMA = 1
    AUTO_ARIMA = 2
    HMM = 3
    GMM_HMM = 4
    M_HMM = 5
    SVM = 6
    SVR = 7
    LIGHT_GBM = 8
    PROPHET = 9


class RunningMode(enum.Enum):
    TRAIN = 1
    PREDICT = 2
    TRAIN_AND_PREDICT = 3

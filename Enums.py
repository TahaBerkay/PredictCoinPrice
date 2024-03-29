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
    CATBOOST = 12
    GHMM = 13
    SVR = 15
    KNN_REGRESSOR = 16
    LINEAR_REGRESSOR = 17
    RANDOMFOREST_REGRESSOR = 18
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


class CurrencySymbols(enum.Enum):
    BTCUSDT = 0
    ETHUSDT = 1
    XRPUSDT = 2
    BCHUSDT = 3
    BSVUSDT = 4
    LTCUSDT = 5
    BNBUSDT = 6
    EOSUSDT = 7
    ADAUSDT = 8
    XTZUSDT = 9
    CROUSDT = 10
    XLMUSDT = 11
    LINKUSDT = 12
    LEOUSDT = 13
    XMRUSDT = 14
    TRXUSDT = 15
    HTUSDT = 16
    NEOUSDT = 17
    ETCUSDT = 18
    USDCUSDT = 19
    DASHUSDT = 20
    MIOTAUSDT = 21
    ATOMUSDT = 22
    MKRUSDT = 23
    VETUSDT = 24
    HEDGUSDT = 25
    ZECUSDT = 26
    XEMUSDT = 27
    ONTUSDT = 28

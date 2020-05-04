# Documentation


## How to run executable with params

```
executable 2 1,2,3 5 MIN_ONE Binance-BTCUSDT-1m.csv
```

Argument | Explanation | Corresponds
------------- | ------------- | -------------
2  | RunningMode | PREDICT
1,2,3  | Comma separated Methods | AUTO_ARIMA, HMM and LIGHT_GBM
5  | Number of future steps | Predict next 5 values
MIN_ONE  | Input intervals | Binance inputs min by min

See the enum list-ids below:

```
class Decision(enum.Enum):
    BUY = 1
    HOLD = 2
    SELL = 3


class Method(enum.Enum):
    AUTO_ARIMA = 1
    HMM = 2
    LIGHT_GBM = 3
    PROPHET = 4
    SVM = 5
    SVR = 6


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

```

pyinstaller --onefile --hidden-import='pkg_resources.py2_warn' --hidden-import='_sysconfigdata_m_linux_x86_64-linux-gnu' --hidden-import='fbprophet' --additional-hooks-dir=PyInstallerHooks Main.py


pip freeze > requirements.txt


pip install -r requirements.txt
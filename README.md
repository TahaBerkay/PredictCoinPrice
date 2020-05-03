# Documentation


## How to run executable with params

```
executable 2 9,1 5
```

Argument | Explanation | Corresponds
------------- | ------------- | -------------
2  | RunningMode | PREDICT
9,1  | Comma separated Methods | PROPHET and ARIMA
5  | Number of future steps | Predict next 5 values

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
```

pyinstaller --onefile --hidden-import='pkg_resources.py2_warn' --hidden-import='_sysconfigdata_m_linux_x86_64-linux-gnu' --hidden-import='fbprophet' --additional-hooks-dir=PyInstallerHooks Main.py
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
```


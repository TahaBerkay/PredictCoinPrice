import datetime
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from MlMethods.Methods import PredictionResult

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd

import CustomSettings
from Enums import Method, RunningMode, DataInterval
from MlMethods import ArimaMethods, HmmMethods, SvmMethods, FacebookMethods, GbmMethods, RnnMethods, KNearestMethods, \
    LinearMethods, RandomForestMethods, NaiveBayesMethods

method_mapping = {
    Method.AUTO_ARIMA: ArimaMethods.AutoArimaMethod,
    Method.HMM: HmmMethods.GaussianHmmMethod,
    Method.LIGHT_GBM: GbmMethods.LightGbmMethod,
    Method.XGBOOST: GbmMethods.XGBoostMethod,
    Method.PROPHET: FacebookMethods.ProphetMethod,
    Method.SVM: SvmMethods.SvmMethod,
    Method.SVR: SvmMethods.SvrMethod,
    Method.KNN_CLASSIFIER: KNearestMethods.KNeighborsClassifierMethod,
    Method.KNN_REGRESSOR: KNearestMethods.KNeighborsRegressorMethod,
    Method.LOGISTIC_REGRESSOR: LinearMethods.LogisticRegressionMethod,
    Method.LINEAR_REGRESSOR: LinearMethods.LinearRegressionMethod,
    Method.RANDOMFOREST_CLASSIFIER: RandomForestMethods.RandomForestClassifierMethod,
    Method.RANDOMFOREST_REGRESSOR: RandomForestMethods.RandomForestRegressorMethod,
    Method.GAUSSIAN_NB: NaiveBayesMethods.GaussianNBMethod,
    Method.BERNOULLI_NB: NaiveBayesMethods.BernoulliNBMethod,
    Method.CATBOOST: GbmMethods.CatBoostMethod,
    Method.GHMM: HmmMethods.GHmmMethod,
    Method.VANILLA_LSTM: RnnMethods.VanillaLstm,
    Method.STACKED_LSTM: RnnMethods.StackedLstm,
    Method.BIDIRECTIONAL_LSTM: RnnMethods.BidirectionalLstm,
    Method.CNN_LSTM: RnnMethods.CnnLstm,
    Method.CONV_LSTM: RnnMethods.ConvLstm,
}


def get_cmd_arguments():
    global running_mode, methods, path
    running_mode = RunningMode[sys.argv[1]]
    methods = sys.argv[2].split(',')
    data_interval = DataInterval[sys.argv[3]]
    file = sys.argv[4]
    path = Path(__file__).parent / file
    assert os.path.exists(path)
    if running_mode != 2:
        os.makedirs(CustomSettings.DATAFILES_DIR, exist_ok=True)
    return running_mode, methods, data_interval, path


def get_csv_data(file_path):
    dateparse = lambda dates: datetime.datetime.strptime(dates, CustomSettings.DATE_FORMAT)
    return pd.read_csv(file_path, sep=',', parse_dates=['Date'], date_parser=dateparse).fillna(0)


def run_algo_combinations():
    result_dict = {result.model: result.data for result in results}
    for algorithms in CustomSettings.ALGORITHM_COMBINATIONS:
        prediction_of_specific_methods = pd.DataFrame([result_dict[x] for x in algorithms])
        means = round(prediction_of_specific_methods.mean(axis=0)).astype(int)
        results.append(PredictionResult(','.join(algorithms), means))


running_mode, methods, data_interval, path = get_cmd_arguments()
csv_data = get_csv_data(path)

executor = ThreadPoolExecutor()
futures = []

for method in methods:
    method_type = Method(int(method))
    method_object = method_mapping[method_type](csv_data, data_interval)
    future = None
    if running_mode == RunningMode.TRAIN:
        future = executor.submit(method_object.train)
    else:
        future = executor.submit(method_object.predict)
    futures.append(future)

executor.shutdown(wait=True)
results = []
for future in futures:
    result = future.result()
    results.append(result)

if running_mode == RunningMode.TRAIN:
    print("Success")
else:
    run_algo_combinations()
    print('Program result:' + json.dumps([result.__dict__ for result in results]))

# sys.stdout.write(json.dumps(results))
# sys.stdout.flush()
# sys.exit(0)

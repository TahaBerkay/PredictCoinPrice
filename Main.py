import datetime
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import CustomSettings
from Enums import Method, RunningMode
from MlMethods import ArimaMethods, HmmMethods, SvmMethods, FacebookMethods

method_mapping = {
    Method.ARIMA: ArimaMethods.ArimaMethod,
    Method.AUTO_ARIMA: ArimaMethods.AutoArimaMethod,
    Method.HMM: HmmMethods.HmmMethod,
    Method.GMM_HMM: HmmMethods.GmmHmmMethod,
    Method.M_HMM: HmmMethods.MultinomialHmmMethod,
    Method.SVM: SvmMethods.SvmMethod,
    Method.SVR: SvmMethods.SvrMethod,
    # Method.LIGHT_GBM: ,
    Method.PROPHET: FacebookMethods.ProphetMethod,
}

running_mode = RunningMode(int(sys.argv[1]))
methods = sys.argv[2].split(',')
nb_of_steps = int(sys.argv[3])

csv_data = None
if running_mode == RunningMode.TRAIN:
    dateparse = lambda dates: datetime.datetime.strptime(dates, CustomSettings.DATE_FORMAT)
    csv_data = pd.read_csv(CustomSettings.DEFAULT_DATASET, sep=',', parse_dates=['Date'],  # index_col='Date',
                           date_parser=dateparse).fillna(0)

executor = ProcessPoolExecutor()
futures = []

for method in methods:
    method_type = Method(int(method))
    method_object = method_mapping[method_type](csv_data)
    future = None

    if running_mode == RunningMode.TRAIN:
        future = executor.submit(method_object.train())

    elif running_mode == RunningMode.PREDICT:
        future = executor.submit(method_object.predict(nb_of_steps))

    else:
        future = executor.submit(method_object.train_and_predict(nb_of_steps))

    futures.append(future)

executor.shutdown(wait=True)
future.result()

import datetime
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import CustomSettings
from Enums import Method, RunningMode, DataInterval
from MlMethods import ArimaMethods, HmmMethods, SvmMethods, FacebookMethods, GbmMethods, RnnMethods

method_mapping = {
    Method.AUTO_ARIMA: ArimaMethods.AutoArimaMethod,
    Method.HMM: HmmMethods.GaussianHmmMethod,
    Method.SVM: SvmMethods.SvmMethod,
    Method.SVR: SvmMethods.SvrMethod,
    Method.LIGHT_GBM: GbmMethods.LightGbmMethod,
    Method.PROPHET: FacebookMethods.ProphetMethod,
    Method.VANILLA_LSTM: RnnMethods.VanillaLstm,
    Method.STACKED_LSTM: RnnMethods.StackedLstm,
    Method.BIDIRECTIONAL_LSTM: RnnMethods.BidirectionalLstm,
    Method.CNN_LSTM: RnnMethods.CnnLstm,
    Method.CONV_LSTM: RnnMethods.ConvLstm,
}


def get_cmd_arguments():
    global running_mode, methods, nb_of_steps, path
    running_mode = RunningMode(int(sys.argv[1]))
    methods = sys.argv[2].split(',')
    nb_of_steps = int(sys.argv[3])
    data_interval = DataInterval[sys.argv[4]]
    file = sys.argv[5]
    path = Path(__file__).parent / file
    assert os.path.exists(path)
    if running_mode != 2:
        os.makedirs(CustomSettings.DATAFILES_DIR, exist_ok=True)
    return running_mode, methods, nb_of_steps, data_interval, path


def get_csv_data(file_path):
    dateparse = lambda dates: datetime.datetime.strptime(dates, CustomSettings.DATE_FORMAT)
    return pd.read_csv(file_path, sep=',', parse_dates=['Date'], date_parser=dateparse).fillna(0)


running_mode, methods, nb_of_steps, data_interval, path = get_cmd_arguments()
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
        future = executor.submit(method_object.predict, nb_of_steps)
    futures.append(future)

executor.shutdown(wait=True)
results = []
for future in futures:
    result = future.result()
    results.append(result)

if running_mode == RunningMode.TRAIN:
    print("Success")
else:
    print('Program result:' + json.dumps([result.__dict__ for result in results]))

# sys.stdout.write(json.dumps(results))
# sys.stdout.flush()
# sys.exit(0)

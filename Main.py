import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import CustomSettings
from Enums import Method, RunningMode
from MlMethods import ArimaMethods, HmmMethods, SvmMethods, FacebookMethods, GbmMethods

method_mapping = {
    Method.AUTO_ARIMA: ArimaMethods.AutoArimaMethod,
    Method.HMM: HmmMethods.GaussianHmmMethod,
    Method.SVM: SvmMethods.SvmMethod,
    Method.SVR: SvmMethods.SvrMethod,
    Method.LIGHT_GBM: GbmMethods.LightGbmMethod,
    Method.PROPHET: FacebookMethods.ProphetMethod,
}


def get_cmd_arguments():
    global running_mode, methods, nb_of_steps, path
    running_mode = RunningMode(int(sys.argv[1]))
    methods = sys.argv[2].split(',')
    nb_of_steps = int(sys.argv[3])
    file = sys.argv[4]
    path = Path(__file__).parent / file
    assert os.path.exists(path)
    if running_mode != 2:
        os.makedirs(CustomSettings.DATAFILES_DIR, exist_ok=True)
    return running_mode, methods, nb_of_steps, path


def get_csv_data(file_path):
    dateparse = lambda dates: datetime.datetime.strptime(dates, CustomSettings.DATE_FORMAT)
    return pd.read_csv(file_path, sep=',', parse_dates=['Date'], date_parser=dateparse).fillna(0)


running_mode, methods, nb_of_steps, path = get_cmd_arguments()
csv_data = get_csv_data(path)

executor = ThreadPoolExecutor()
futures = []

for method in methods:
    method_type = Method(int(method))
    method_object = method_mapping[method_type](csv_data)
    future = None

    if running_mode == RunningMode.TRAIN:
        future = executor.submit(method_object.train)

    elif running_mode == RunningMode.PREDICT:
        future = executor.submit(method_object.predict, nb_of_steps)

    else:
        future = executor.submit(method_object.train_and_predict, nb_of_steps)

    futures.append(future)

executor.shutdown(wait=True)
results = []
for future in futures:
    result = future.result()
    results.append(result)

print(json.dumps([result.__dict__ for result in results]))
# sys.stdout.write(json.dumps(results))
# sys.stdout.flush()
# sys.exit(0)

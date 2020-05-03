import abc
import os
import pickle
import time

import CustomSettings

file_mode_write_binary = 'wb'
file_mode_read_binary = 'rb'
file_extension = '.pkl'


class PredictionResult:
    def __init__(self, method_name, data):
        self.model = method_name
        self.data = data.tolist()


class Method(abc.ABC):
    def __init__(self, data):
        self.data = data
        self.model = None

    @abc.abstractmethod
    def fit_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forecast(self, nb_of_steps):
        raise NotImplementedError

    @abc.abstractmethod
    def manipulate_data(self, data):
        raise NotImplementedError

    def save_model(self):
        path = os.path.join(CustomSettings.DATAFILES_DIR, self.__class__.__name__ + file_extension)
        with open(path, file_mode_write_binary) as pkl:
            pickle.dump(self.model, pkl)

    def load_model(self):
        path = os.path.join(CustomSettings.DATAFILES_DIR, self.__class__.__name__ + file_extension)
        with open(path, file_mode_read_binary) as pkl:
            self.model = pickle.load(pkl)

    def modify_result(self, result):
        return PredictionResult(self.__class__.__name__, result)

    def train(self):
        start_time = time.time()
        self.manipulate_data()
        self.fit_model()
        self.save_model()
        print("%s execution time -train-: %s seconds" % (self.__class__.__name__, (time.time() - start_time)))

    def predict(self, nb_of_steps):
        start_time = time.time()
        self.manipulate_data()
        self.load_model()
        prediction = self.forecast(nb_of_steps)
        print("%s execution time -predict-: %s seconds" % (self.__class__.__name__, (time.time() - start_time)))
        return self.modify_result(prediction)

    def train_and_predict(self, nb_of_steps):
        train_start_time = time.time()
        self.manipulate_data()
        self.fit_model()
        self.save_model()
        predict_start_time = time.time()
        print("%s execution time -train-: %s seconds" % (
            self.__class__.__name__, (predict_start_time - train_start_time)))
        prediction = self.forecast(nb_of_steps)
        print("%s execution time -predict-: %s seconds" % (self.__class__.__name__, (time.time() - predict_start_time)))
        return self.modify_result(prediction)

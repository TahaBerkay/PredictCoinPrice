import abc
import os
import pickle

import CustomSettings

file_mode_write_binary = 'wb'
file_mode_read_binary = 'rb'
file_extension = '.pkl'


class PredictionResult:
    def __init__(self, method_name, data):
        self.model = method_name
        self.data = data
        # if isinstance(data, str):  # TODO
        #    self.data = data.split()
        # else:
        #    self.data = data.tolist()


class Method(abc.ABC):
    def __init__(self, data, data_interval):
        self.data = data
        self.data_interval = data_interval
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
        path = os.path.join(CustomSettings.DATAFILES_DIR,
                            self.__class__.__name__ + '_' + self.data_interval.name + file_extension)
        with open(path, file_mode_write_binary) as pkl:
            pickle.dump(self.model, pkl)

    def load_model(self):
        path = os.path.join(CustomSettings.DATAFILES_DIR,
                            self.__class__.__name__ + '_' + self.data_interval.name + file_extension)
        with open(path, file_mode_read_binary) as pkl:
            self.model = pickle.load(pkl)

    def modify_result(self, result):
        return PredictionResult(self.__class__.__name__, result)

    def train(self):
        self.manipulate_data()
        self.fit_model()
        self.save_model()

    def predict(self, nb_of_steps):
        self.manipulate_data()
        self.load_model()
        prediction = self.forecast(nb_of_steps)
        return self.modify_result(prediction)

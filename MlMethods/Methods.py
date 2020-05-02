import abc
import pickle

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
        with open(self.__class__.__name__ + file_extension, file_mode_write_binary) as pkl:
            pickle.dump(self.model, pkl)

    def load_model(self):
        with open(self.__class__.__name__ + file_extension, file_mode_read_binary) as pkl:
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
        return self.modify_result(self.forecast(nb_of_steps))

    def train_and_predict(self, nb_of_steps):
        self.manipulate_data()
        self.fit_model()
        self.save_model()
        return self.modify_result(self.forecast(nb_of_steps))
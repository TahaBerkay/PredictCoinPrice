import os

import pandas as pd
from hmmlearn.hmm import GMMHMM
# from hmmlearn.hmm import MultinomialHMM
from hmmlearn.hmm import GaussianHMM
from pandas import np

import CustomSettings
from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class GaussianHmmMethod(Methods.Method):
    window_size = CustomSettings.WINDOW_SIZE

    def __init__(self, data, data_interval, currency_symbol):
        super().__init__(data, data_interval, currency_symbol)
        self.past_likelihood = []
        self.past_change = []
        self.model_data_path = os.path.join(CustomSettings.DATAFILES_DIR,
                                            self.currency_symbol.name + '_' + self.data_interval.name + '_' + self.__class__.__name__ + '_model_data' + '.pkl')

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        nb_of_states = 20  # self.get_number_of_opt_states()
        model = GaussianHMM(n_components=nb_of_states, covariance_type='full', tol=0.0001, n_iter=10000)
        self.model = model.fit(data)
        start_idx = 0
        last_interval_start_idx = data.shape[0] - self.window_size - 1
        while start_idx < last_interval_start_idx:
            end_idx = start_idx + self.window_size
            self.past_likelihood = np.append(self.past_likelihood,
                                             model.score(data.iloc[start_idx:end_idx, :]))
            self.past_change = np.append(self.past_change,
                                         (pd.np.sign(
                                             self.data['Close'][end_idx] - self.data['Close'][end_idx - 1]) + 1))
            start_idx += 1
        self.past_change = self.past_change.astype(int)

    def forecast(self):
        self.data = DatasetProcessor.preprocess_input_data(self.data)
        curr_likelihood = self.model.score(self.data.tail(self.window_size))
        likelihood_diff_idx = np.argmin(np.absolute(self.past_likelihood - curr_likelihood))
        predicted_change = self.past_change[likelihood_diff_idx]
        return predicted_change

    def get_number_of_opt_states(self):
        bic_vect = np.empty([0, 1])
        for states in range(2, 20):
            num_params = states ** 2 + states
            model = GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=10000)
            model.fit(self.data)
            bic_vect = np.vstack((bic_vect, -2 * model.score(self.data) + num_params * np.log(self.data.shape[0])))
        return np.argmin(bic_vect) + 2

    def save_model(self):
        super().save_model()
        model_data = pd.DataFrame({'likelihood': self.past_likelihood, 'change': self.past_change})
        model_data.to_pickle(self.model_data_path)

    def load_model(self):
        super().load_model()
        model_data = pd.read_pickle(self.model_data_path)
        self.past_change = model_data['change']
        self.past_likelihood = model_data['likelihood']


class GHmmMethod(Methods.Method):
    window_size = CustomSettings.WINDOW_SIZE

    def __init__(self, data, data_interval, currency_symbol):
        super().__init__(data, data_interval, currency_symbol)
        self.model_data_path = os.path.join(CustomSettings.DATAFILES_DIR,
                                            self.currency_symbol.name + '_' + self.data_interval.name + '_' + self.__class__.__name__ + '_model_data' + '.pkl')

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        nb_of_states = 5  # self.get_number_of_opt_states()

        sell_indexes = [i for i in range(len(label)) if label[i] == 0]
        sell_model = GaussianHMM(n_components=nb_of_states, covariance_type='full', tol=0.0001, n_iter=10000)
        sell_model = sell_model.fit(data.iloc[sell_indexes, :])

        hold_indexes = [i for i in range(len(label)) if label[i] == 1]
        hold_model = GaussianHMM(n_components=nb_of_states, covariance_type='full', tol=0.0001, n_iter=10000)
        hold_model = hold_model.fit(data.iloc[hold_indexes, :])

        buy_indexes = [i for i in range(len(label)) if label[i] == 2]
        buy_model = GaussianHMM(n_components=nb_of_states, covariance_type='full', tol=0.0001, n_iter=10000)
        buy_model = buy_model.fit(data.iloc[buy_indexes, :])

        self.model = dict({"sell_model": sell_model, "hold_model": hold_model, "buy_model": buy_model})

    def forecast(self):
        self.data = DatasetProcessor.preprocess_input_data(self.data)
        log_likelihoods = (self.model["sell_model"].score(self.data.tail(1)),
                           self.model["hold_model"].score(self.data.tail(1)),
                           self.model["buy_model"].score(self.data.tail(1)))
        prediction = np.where(log_likelihoods == np.amax(log_likelihoods))
        return prediction[0]

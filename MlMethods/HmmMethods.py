import os

import pandas as pd
# from hmmlearn.hmm import GMMHMM
# from hmmlearn.hmm import MultinomialHMM
from hmmlearn.hmm import GaussianHMM
from pandas import np

import CustomSettings
from MlMethods import Methods


class GaussianHmmMethod(Methods.Method):
    interval = 30

    def __init__(self, data):
        super().__init__(data)
        self.past_likelihood = []
        self.model_data_path = os.path.join(CustomSettings.DATAFILES_DIR, self.__class__.__name__ + '_model_data.pkl')
        self.past_likelihood_path = os.path.join(CustomSettings.DATAFILES_DIR,
                                                 self.__class__.__name__ + '_past_likelihood.npy')

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        nb_of_states = self.get_number_of_opt_states()
        model = GaussianHMM(n_components=nb_of_states, covariance_type='full', tol=0.0001, n_iter=10000)
        self.model = model.fit(self.data)

        iters = 0;
        last_interval_start_idx = self.data.shape[0] - self.interval - 1
        while iters < last_interval_start_idx:
            self.past_likelihood = np.append(self.past_likelihood,
                                             model.score(self.data.iloc[iters:iters + self.interval, :]))
            iters += 1

        self.save_data()

    def forecast(self, nb_of_steps):
        self.load_data()
        curr_likelihood = self.model.score(self.data.tail(self.interval))

        likelihood_diff_idx = np.argmin(np.absolute(self.past_likelihood - curr_likelihood))
        last_index_of_selected_frame = likelihood_diff_idx + self.interval - 1

        predicted_change = self.data['Close'][last_index_of_selected_frame] - self.data['Close'][
            last_index_of_selected_frame + 1]
        return self.data.iloc[-1]['Close'] + predicted_change

    def get_number_of_opt_states(self):
        bic_vect = np.empty([0, 1])
        for states in range(3, 8):
            num_params = states ** 2 + states
            model = GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=10000)
            model.fit(self.data)
            bic_vect = np.vstack((bic_vect, -2 * model.score(self.data) + num_params * np.log(self.data.shape[0])))
        return np.argmin(bic_vect) + 2

    def save_data(self):
        self.data.to_pickle(self.model_data_path)
        np.save(self.past_likelihood_path, self.past_likelihood)

    def load_data(self):
        self.data = pd.read_pickle(self.model_data_path)
        self.past_likelihood = np.load(self.past_likelihood_path)

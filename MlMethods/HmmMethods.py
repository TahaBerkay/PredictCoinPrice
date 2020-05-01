from hmmlearn.hmm import GMMHMM
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM

from MlMethods import Methods


class HmmMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        self.model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
        self.model.fit(self.data)

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(self.data)
        return predictions


class GmmHmmMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        self.model = GMMHMM(n_components=4, n_mix=6, n_iter=1000)
        self.model.fit(self.data)

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(self.data)
        return predictions


class MultinomialHmmMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        self.model = MultinomialHMM(n_components=4, n_iter=1000)
        self.model.fit(self.data)

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(self.data)
        return predictions

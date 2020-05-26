from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class GaussianNBMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GaussianNB()
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction


class BernoulliNBMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = BernoulliNB()
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction

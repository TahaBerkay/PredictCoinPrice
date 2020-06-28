from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import CustomSettings
from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class LogisticRegressionMethod(Methods.Method):
    grid_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid=self.grid_params,
            cv=4,
            n_jobs=CustomSettings.NB_JOBS_GRIDSEARCH,
            scoring='balanced_accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction


class LinearRegressionMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = LinearRegression()
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction

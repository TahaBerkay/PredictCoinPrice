from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

import CustomSettings
from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods

grid_params = {'leaf_size': range(2, 40, 2),
               'n_neighbors': range(2, 15),
               'p': [1, 2]}


class KNeighborsClassifierMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=KNeighborsClassifier(n_jobs=CustomSettings.NB_JOBS_GRIDSEARCH),
            param_grid=grid_params,
            cv=4,
            scoring='balanced_accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction


class KNeighborsRegressorMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=KNeighborsRegressor(n_jobs=CustomSettings.NB_JOBS_GRIDSEARCH),
            param_grid=grid_params,
            cv=4,
            scoring='neg_root_mean_squared_error',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction

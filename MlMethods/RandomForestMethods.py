from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import CustomSettings
from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods

grid_params = {'max_depth': [None, 3, 5, 7, 9, 11],
               'n_estimators': range(80, 220, 20)}


class RandomForestClassifierMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=grid_params,
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


class RandomForestRegressorMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=grid_params,
            cv=4,
            n_jobs=CustomSettings.NB_JOBS_GRIDSEARCH,
            scoring='neg_root_mean_squared_error',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import SVR

from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods

grid_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': ['scale', 'auto'],
               'kernel': ['linear', 'rbf']}


class SvmMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=SVC(),
            param_grid=grid_params,
            cv=4,
            n_jobs=-1,
            scoring='accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(data.tail(1))
        return prediction


class SvrMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=SVR(),
            param_grid=grid_params,
            cv=4,
            n_jobs=-1,
            scoring='accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        data = DatasetProcessor.preprocess_input_data(self.data)
        predictions = self.model.predict(data.tail(1))
        return predictions

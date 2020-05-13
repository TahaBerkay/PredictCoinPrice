from pandas import np
from sklearn.svm import SVC
from sklearn.svm import SVR

from Enums import Decision
from MlMethods import Methods

grid_params = {'C': [0.001, 0.01, 0.1],
               'gamma': ['scale', 'auto'],
               'kernel': ['linear', 'rbf']}


class SvmMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data_labels = np.sign(np.diff(self.data['Close'].to_numpy())) + 1
        self.data = self.data[: -1]
        # self.model = GridSearchCV(estimator=SVC(), param_grid=grid_params)
        self.model = SVC()
        self.model.fit(self.data, data_labels)

    def forecast(self, nb_of_steps):
        predictions = self.model.predict(self.data.iloc[-1].to_numpy().reshape(1, -1))
        return Decision(predictions).name


class SvrMethod(Methods.Method):

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data_labels = self.data['Close'].shift(-1)[:-1]
        self.data = self.data[: -1]
        # self.model = GridSearchCV(estimator=SVR(), param_grid=grid_params)
        self.model = SVR()
        self.model.fit(self.data, data_labels)

    def forecast(self, nb_of_steps):
        predictions = self.model.predict(self.data.iloc[-1].to_numpy().reshape(1, -1))
        return predictions[0]

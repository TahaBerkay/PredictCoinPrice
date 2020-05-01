from sklearn.svm import SVC
from sklearn.svm import SVR

from MlMethods import Methods


class SvmMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        diff = self.data.shift(-1) - self.data
        diff[diff >= 0] = 1
        diff[diff < 0] = 0
        self.model = SVC(kernel='rbf')
        self.model.fit(self.data, diff)
        # nus = [0.00000006, 0.00000001, 0.0000006, 0.0000001, 0.000006, 0.000001, 0.00006, 0.00001, 0.0006, 0.0001, 0.006, 0.001]
        # gammas = [0.00000006, 0.00000001, 0.0000006, 0.0000001, 0.000006, 0.000001, 0.00006, 0.00001, 0.0006, 0.0001, 0.006, 0.001]
        # clf = GridSearchCV(svm.OneClassSVM(), [{'kernel': ['rbf'], 'nu': nus, 'gamma': gammas}], cv=3, scoring='recall')
        # clf.fit(train_x, train_x.shape[0] * [1])

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(self.data[-1])
        return predictions


class SvrMethod(Methods.Method):
    def __init__(self, data):
        super().__init__(data)

    def fit_model(self):
        diff = self.data.shift(-1) - self.data
        diff[diff >= 0] = 1
        diff[diff < 0] = 0
        self.model.fit(self.data, diff)
        # nus = [0.00000006, 0.00000001, 0.0000006, 0.0000001, 0.000006, 0.000001, 0.00006, 0.00001, 0.0006, 0.0001, 0.006, 0.001]
        # gammas = [0.00000006, 0.00000001, 0.0000006, 0.0000001, 0.000006, 0.000001, 0.00006, 0.00001, 0.0006, 0.0001, 0.006, 0.001]
        # clf = GridSearchCV(svm.OneClassSVM(), [{'kernel': ['rbf'], 'nu': nus, 'gamma': gammas}], cv=3, scoring='recall')
        # clf.fit(train_x, train_x.shape[0] * [1])
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
        svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
        svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
        ###########################################
        # svr_rbf.fit(dates, prices)  # fitting the data points in the models
        # svr_lin.fit(dates, prices)
        # svr_poly.fit(dates, prices)

    def forecast(self, nbOfSteps):
        predictions = self.model.predict(self.data[-1])
        return predictions

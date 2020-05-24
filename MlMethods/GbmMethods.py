import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class LightGbmMethod(Methods.Method):
    params = [{'max_depth': [-1]},
              {'num_leaves': [4, 5, 6], 'max_depth': [3], 'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]},
              {'num_leaves': [8, 10, 12, 14], 'max_depth': [4], 'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]},
              {'num_leaves': [16, 20, 24, 28], 'max_depth': [5], 'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]},
              {'num_leaves': [32, 40, 48, 56], 'max_depth': [6], 'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]},
              {'num_leaves': [64, 80, 96, 112], 'max_depth': [7], 'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]},
              {'num_leaves': [128, 160, 192, 224], 'max_depth': [8],
               'min_data_in_leaf': [25, 50, 75, 100, 125, 150, 175]}]

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=lgb.LGBMClassifier(),
            param_grid=self.params,
            cv=4,
            n_jobs=-1,
            scoring='accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        input = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(input)
        return prediction[-1]

    def feature_importance(self):
        print('Feature importances:', list(self.model.feature_importances_))
        # plt.show()


class XGBoostMethod(Methods.Method):
    params = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 12, 2),
        'gamma': [0.5, 0, 1, 1.5, 2, 5, 7, 9],
        # 'n_estimators': [60, 80, 100, 120, 140, 160],
        # 'subsample': [0.5,1]
    }

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = GridSearchCV(
            estimator=xgb.XGBClassifier(),
            param_grid=self.params,
            cv=4,
            n_jobs=-1,
            scoring='accuracy',
            verbose=2
        )
        self.model.fit(data, label)

    def forecast(self):
        input = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(input)
        return prediction[-1]

    def feature_importance(self):
        print('Feature importances:', list(self.model.feature_importances_))
        # plt.show()

import lightgbm as lgb

from DatasetHandler.DatasetProcessor import DatasetProcessor
from MlMethods import Methods


class LightGbmMethod(Methods.Method):
    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        data = DatasetProcessor.preprocess_input_data(self.data)[:-1]
        label = DatasetProcessor.prepare_labels(self.data)
        self.model = lgb.LGBMClassifier()
        self.model.fit(data, label)

    def forecast(self, nb_of_steps):
        input = DatasetProcessor.preprocess_input_data(self.data)
        prediction = self.model.predict(input)
        return prediction[-1]

    def feature_importance(self):
        print('Feature importances:', list(self.model.feature_importance()))
        # plt.show()

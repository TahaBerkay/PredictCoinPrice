import lightgbm as lgb

from MlMethods import Methods


class LightGbmMethod(Methods.Method):
    params = {
        "objective": "mape",
        "num_leaves": 124,
        "min_data_in_leaf": 340,
        "learning_rate": 0.1,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.87,
        "bagging_freq": 19,
        "num_rounds": 940,
        "early_stopping_rounds": 125,
        "num_threads": 16,
        "seed": 1,
    }

    def manipulate_data(self):
        self.data = self.data.drop("Date", axis=1, inplace=False)

    def fit_model(self):
        dataset = lgb.Dataset(self.data.drop("Close", axis=1, inplace=False)[:-1], label=self.data["Close"].shift(-1)[:-1])
        self.model = lgb.train(self.params, dataset, valid_sets=[dataset], verbose_eval=50)

    def forecast(self, nb_of_steps):
        prediction = self.model.predict(self.data.drop("Close", axis=1, inplace=False).iloc[-1])
        return prediction

    def feature_importance(self):
        print('Feature importances:', list(self.model.feature_importance()))
        # plt.show()

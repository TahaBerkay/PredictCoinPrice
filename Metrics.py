import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, \
    jaccard_score, \
    balanced_accuracy_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    classification_report


class AccuracyResult:
    def __init__(self, name, accuracy_score, \
                 balanced_accuracy_score, \
                 jaccard_score_micro, \
                 jaccard_score_macro, \
                 f1_score_micro, \
                 f1_score_macro, \
                 precision_score_micro, \
                 precision_score_macro, \
                 recall_score_micro, \
                 recall_score_macro):
        self.name = name
        self.accuracy_score = accuracy_score
        self.balanced_accuracy_score = balanced_accuracy_score
        self.jaccard_score_micro = jaccard_score_micro
        self.jaccard_score_macro = jaccard_score_macro
        self.f1_score_micro = f1_score_micro
        self.f1_score_macro = f1_score_macro
        self.precision_score_micro = precision_score_micro
        self.precision_score_macro = precision_score_macro
        self.recall_score_micro = recall_score_micro
        self.recall_score_macro = recall_score_macro


all_result_map = {}

directory = r'ModelResults'
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        path = os.path.join(directory, filename)
        with open(path) as json_file:
            data = json.load(json_file)
            df = pd.DataFrame(data, columns=['true', 'prediction'])
            all_result_map[os.path.splitext(filename)[0]] = df
            accuracy_result = AccuracyResult(filename,
                                             accuracy_score(df['true'], df['prediction']),
                                             balanced_accuracy_score(df['true'], df['prediction']),
                                             jaccard_score(df['true'], df['prediction'], average='micro'),  # weighted
                                             jaccard_score(df['true'], df['prediction'], average='macro'),
                                             f1_score(df['true'], df['prediction'], average='micro'),
                                             f1_score(df['true'], df['prediction'], average='macro'),
                                             precision_score(df['true'], df['prediction'], average='micro'),
                                             precision_score(df['true'], df['prediction'], average='macro'),
                                             recall_score(df['true'], df['prediction'], average='micro'),
                                             recall_score(df['true'], df['prediction'], average='macro')
                                             )

            print('Accuracy result:' + json.dumps([accuracy_result.__dict__]))
            #print(classification_report(df['true'], df['prediction']))
    else:
        continue

methods = ['LightGbmMethod',
          'CatBoostMethod',
          'GaussianNBMethod',
          'BernoulliNBMethod',
          'LogisticRegressionMethod',
          'SvmMethod',
          'XGBoostMethod',
          'GHmmMethod']

max_acc = {}


x = len(methods)
for idx, i in enumerate(range(1 << x)):
    print(idx)
    subset = [methods[j] for j in range(x) if (i & (1 << j))]
    if len(subset) == 0:
        continue
    prediction_of_specific_methods = pd.DataFrame([all_result_map[x]['prediction'] for x in subset])
    means = round(prediction_of_specific_methods.mean(axis=0)).astype(int)
    most_frequents = prediction_of_specific_methods.apply(lambda x: x.value_counts().idxmax())

    #print(subset)
    #print('--1--')
    #print(classification_report(df['true'], means))

    #print('--2--')
    #print(classification_report(df['true'], most_frequents))

    #max_acc[', '.join(map(str, subset))] = max(accuracy_score(df['true'], means),
    #                                           accuracy_score(df['true'], most_frequents))

    max_acc[', '.join(map(str, subset)) + ':' + 'mean'] = accuracy_score(df['true'], means)
    max_acc[', '.join(map(str, subset)) + ':' + 'most_frequents'] = accuracy_score(df['true'], most_frequents)


    #print('****************************')

max_key = max(max_acc, key=max_acc.get)
print(max_key)
print(max_acc[max_key])

sorted_dict = sorted(max_acc, key=max_acc.get, reverse=True)
print(sorted_dict[9])
print(max_acc[sorted_dict[9]])

print("\n".join(sorted_dict[:10]))
#print("\n".join(max_acc[sorted_dict[:10]]))   # matrix = [[j for j in range(5)] for i in range(5)]


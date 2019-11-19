import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import statsmodels.api as sm

warnings.filterwarnings("ignore")


train_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
test_set1_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
test_set2_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test2.csv', index_col=0)


concanate_df = pd.concat([train_df,test_set1_df,test_set2_df])
print("Combine df shape: ",concanate_df.shape)
del train_df,test_set1_df,test_set2_df

numeric_df = concanate_df.select_dtypes(include=[np.number])
print("Combine numeric df shape: ",numeric_df.shape)

rents = numeric_df['rent']

all_value_df = numeric_df.dropna(axis=1)
all_value_df['rent'] = rents
print("Combine no null value numeric df shape: ",all_value_df.shape)

new_df = all_value_df.iloc[:14000] # combin train and test 1 set

columns = list(all_value_df.columns)
columns.append('const')
new_df = sm.add_constant(new_df)
est = sm.OLS(new_df['rent'],
             new_df[columns].astype(float)).fit()
print(est.summary())

feature_columns = ['bedrooms','bathrooms','size_sqft','addr_zip','has_elevator','has_fireplace','has_dishwasher','is_furnished','has_gym','allows_pets','has_washer_dryer','has_roofdeck','has_concierge','no_fee']


# features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
#             'emb_S', 'emb_C', 'emb_Q', 'is_male', 'has_cabin']
#
# valid = df[features].notna().all(axis=1)
# print(len(df), sum(valid))

dtree=DecisionTreeClassifier(
    criterion='entropy',
    random_state=20191021,
    max_depth=5,
    #min_samples_split=2,
    #min_samples_leaf=1,
    #max_features=None,
    #max_leaf_nodes=None,
)

dtree.fit(df[features], df['Survived'])

#Visualization
dot_data = StringIO()
export_graphviz(dtree,
                out_file=dot_data,
                filled=True,
                rounded=True,
                feature_names=features,
                special_characters=True
               )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

pred_survival = dtree.predict(df[features])

print(confusion_matrix(df.Survived, pred_survival), '\n')
print('Accuracy:   %0.3f' % accuracy_score(df.Survived, pred_survival))
print('Precision:  %0.3f' % precision_score(df.Survived, pred_survival))
print('Recall:     %0.3f' % recall_score(df.Survived, pred_survival))

#Cross Validation
k_fold = KFold(n_splits=5, random_state=20181105,shuffle=False)
for train_indices, test_indices in k_fold.split(df[features]):
     print('Train: n=%i, s_rate=%0.2f | test: n=%i, s_rate=%0.2f ' %
           (df.loc[train_indices, 'Survived'].count(),
            df.loc[train_indices, 'Survived'].mean(),
            df.loc[test_indices, 'Survived'].count(),
            df.loc[test_indices, 'Survived'].mean(),
           )
          )

# Test for CV
def get_cv_results(classifier):
    results = []
    for train, test in k_fold.split(df[features]):
        classifier.fit(df.loc[train, features], df.loc[train, 'Survived'])
        y_predicted = classifier.predict(df.loc[test, features])
        accuracy = accuracy_score(df.loc[test, 'Survived'], y_predicted)
        results.append(accuracy)

    return np.mean(results), np.std(results)


hp_values = range(10, 200, 10)
# m_values = range(10,200,20)
all_mu = []
all_sigma = []

for m in hp_values:
    dtree = DecisionTreeClassifier(
        criterion='entropy',
        random_state=20180408,
        min_samples_split=m,
        max_depth=m,
        # min_samples_leaf=m,
        # max_features=m,
        # max_leaf_nodes=m,
    )

    mu, sigma = get_cv_results(dtree)
    all_mu.append(mu)
    all_sigma.append(sigma)

    print(m, mu, sigma)
plt.figure(figsize=(14, 5))
plt.plot(hp_values, all_mu)
plt.ylabel('Cross Validation Accuracy')
plt.xlabel('Minimum Samples Per Leaf')

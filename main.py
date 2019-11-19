from Helper.DataLoader import *
from Helper.DataVisualization import statistic_analysis
from Helper.SearchCVModels import CV_Model
from Helper.StaticParameters import Parameters

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

parameter = Parameters()

combine_df, train_df_index, test_set1_index, test_set2_index = load_dataset(numeric=True, extract_dataset=False)
combine_df = clean_data(combine_df)
combine_df = normalized_dataset(combine_df)

DTree = DecisionTreeClassifier()
SearchCV = CV_Model(GridSearch=False)
SearchCV.train_model(DTree, parameter.dtree_parameter, combine_df.iloc[:12000], parameter.feature_columns, cv_split = 6)
SearchCV.test_model(combine_df.iloc[12000:14000], parameter.feature_columns)
# statistic_analysis(combine_df)

KNN = KNeighborsRegressor()
SearchCV = CV_Model(GridSearch=False)
SearchCV.train_model(KNN, parameter.knn_parameter, combine_df.iloc[:12000], parameter.feature_columns, cv_split = 6)
SearchCV.test_model(combine_df.iloc[12000:14000], parameter.feature_columns)
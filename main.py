from Helper.DataLoader import *
from Helper.StaticParameters import Parameters
from Helper.ModelProcessor import ModelProcessor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

# Init model
parameter = Parameters()
modelProcessor = ModelProcessor()

# Get Data
combine_df, train_df_index, test_set1_index, test_set2_index = load_dataset(numeric=True, extract_dataset=False)
combine_df = clean_data(combine_df)
combine_df = normalized_dataset(combine_df)

# Train Models - All Selected Features
modelProcessor.train_model(classifier=LinearRegression(),parameters=parameter.linreg_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:12000], test_df=combine_df.iloc[12000:14000],
                           grid=True,train_target='rent',cv_split = 4, trainWithTest1=False,
                           test2_df=combine_df.iloc[14000:])
#Decision Tree
modelProcessor.train_model(classifier=DecisionTreeRegressor(),parameters=parameter.dtree_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)


#KNN
modelProcessor.train_model(classifier=KNeighborsRegressor(),parameters=parameter.knn_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)

#Linear Regression
modelProcessor.train_model(classifier=LinearRegression(),parameters=parameter.linreg_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)

#Gradient Boosting
modelProcessor.train_model(classifier=GradientBoostingRegressor(loss="ls"),parameters=parameter.gradient_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)


# Ada Boosting with Linear Regression
modelProcessor.train_model(classifier=AdaBoostRegressor(base_estimator=LinearRegression()),
                           parameters=parameter.ada_parameter,
                           feature_columns= parameter.feature_columns,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)


# Train Model - Less Features
features = ['bedrooms', 'bathrooms', 'size_sqft', 'addr_zip', 'floor_count', 'min_to_subway', 'has_doorman',
                        'is_furnished', 'allows_pets', 'no_fee','floornumber']

#Decision Tree
modelProcessor.train_model(classifier=DecisionTreeRegressor(),parameters=parameter.dtree_less_parameter,
                           feature_columns= features,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)


#KNN
modelProcessor.train_model(classifier=KNeighborsRegressor(),parameters=parameter.knn_parameter,
                           feature_columns= features,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)

#Linear Regression
modelProcessor.train_model(classifier=LinearRegression(),parameters=parameter.linreg_parameter,
                           feature_columns= features,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)

#Gradient Boosting
modelProcessor.train_model(classifier=GradientBoostingRegressor(loss="ls"),parameters=parameter.gradient_parameter,
                           feature_columns= features,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)


# Ada Boosting with Linear Regression
modelProcessor.train_model(classifier=AdaBoostRegressor(base_estimator=LinearRegression()),
                           parameters=parameter.ada_parameter,
                           feature_columns= features,
                           train_df= combine_df.iloc[:14000], test_df=combine_df.iloc[14000:],
                           grid=False,train_target='rent',cv_split = 4)
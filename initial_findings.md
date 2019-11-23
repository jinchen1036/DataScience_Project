## Strategies 
#### Feauture Extraction Technique
- Understand the Data (Reference to Practice Analysis Folder)
    - Ignore all object fields and focus on the numeric fields.  (Will do further analysis on object fields in Part II)
    - Visualize all the numeric fields that is not boolean value (FeatureAnalysis.ipynb)
        - Identify the outliers and see if the data are correct
        - Identify the fields that need to update the outliers and fill the na values 
- Clean the Data (DataLoader.py)
    - Combine the train data with test data set 1 and 2, and clean it in the same way
        - This will minimize the error of replace the error value, as they will be replace as same value for all data
        - Also as the data are from same source, each field should be have similar characterize.
    - Use median value of each field to replace the error/na value
        - Median is better than mean value, since outliers can have big influence in the mean value
    - Normalize the clean dataset, so each field are arrange from 0 to 1
        - Make the influence of each field equally distribute to the model.
- Statistic Analysis identify which fields to be used for train the model
    - Using Ordinary Least Squares to see the relation of each field with the target (rent)
    - Use the fields that with p-value equal to 0 
    
#### Model Selection and Training
- Find the best hyperparameters of each model
    - GridSearchCV -> decide the final hyperparameters
    - RandomizedSearchCV -> for quite analysis purpose
- Try Various Models for Regression Problem and find the best one
    - List of Models Tried
        - LinearRegression
        - KNeighborsRegressor
        - DecisionTreeRegressor
        - GradientBoostingRegressor
        - AdaBoostRegressor
        - RandomForestRegressor
    - Train with train dataset, and see the loss of both test set 1 and 2
        - using the median of train dataset rent as the true label for determine the test set 2 loss
        - Find the model with smallest loss for test set 1 and retrain with both train and test data set 1 to get the final model
        - Using the final model to predict the test set 2 rent

#### Improvement for Part II
- Using other dataset, such as income or population around the neigborhood.
- Using the object fields, treat as categories and turn into integer
- Combine different model predictions and average the prediction to get the final result


## Model Performance
To find the best model with best performace, our approach was as followed:
1. Conduct EDA on the three given data sets (training set, test1 set, test2 set) by visualizing all the numerical fields that are not boolean (binary) values using histograms. We picked out the obvious outliers and see if they are truly outliers (i.e. if the value makes sense) by getting the min and max of the columns. We also identified the fields that contain null values.
2. Clean the data based on what we found during the EDA:
    - Combine the training data with test data set 1 and 2, and clean them together in the same way as they are from the same source and each field should have similar chracterisitcs.
    - Use median value of each field to replace the outlier/null values. We used median instead of mean because outliers could have big influence on the mean.
    - Normalize the cleand dataset so that the range of each field is from 0 to 1. This can make sure the influence of each field is equally distributed to the model.
3. Conduct statistic analysis to identify which fields to be used for training the model:
    - Use Ordinary Least Squares to see the relation of each field with the target (rent).
    - Use the fields with p-value equal to 0.
4. Try various models for regression problem and find the best one. As for hyperparamters, at first we used RandomizedSearchCV to randomize the hyperparameters for quick analysis purposes. Then we used GridSeachCV to find the best hyperparameters.
    - List of models that we have tried are:
        - LinearRegression
        - KNeighborsRegressor
        - DecisionTreeRegressor
        - GradientBoostingRegressor
        - AdaBoostRegressor (results were not good)
        - RandomForestRegressor
    - Train the models with training dataset, and calculate the loss of both test set 1 and 2:
        - Use the median of rent in training dataset as the true label for determining the test set 2 loss
        - Find the model with smallest loss for test set 1, and re-train that model with both training set and test set 1 to get the final model
        - Use the final model to predict the test set 2 rent

more features = ['bedrooms', 'bathrooms', 'size_sqft', 'addr_zip', 'floor_count', 'min_to_subway', 'has_doorman', 
                                'has_elevator', 'has_dishwasher', 'is_furnished', 'has_gym', 'allows_pets', 'has_washer_dryer', 
                                'has_concierge', 'no_fee', 'has_pool', 'floornumber']

less features = ['bedrooms', 'bathrooms', 'year_built', 'addr_zip', 'bathrooms',' size_sqft', 'no_fee']

This table contains the performance of various models we tried in terms of mean squared error, they were compared with using more features and less features.

|                   |       more features         ||         less features      ||
|-------------------|--------------|--------------|--------------|--------------|
|                   | test1 loss   | test2 loss   | test1 loss   | test2 loss   |
| Linear Regression | 2870402.2234 | 5830874.4832 | 3130781.1625 | 5702636.3532 |
|   Decision Tree   | 2853766.8381 | 6596728.8758 |              |              |
|     KNeighbors    | 3911766.0539 | 3723764.7534 | 2305387.7376 | 6478483.8429 |
| Gradient Boosting | 4865627.4079 | 1133123.8333 | 4874866.8723 | 1160366.4162 |
|   Random Forest   | 1737228.7359 | 6569618.5177 | 1787680.1939 | 6790996.1342 |

Based on the comparison of these models, we decided the final model to be Random Forest. Its performance were compared with using more features, less features, as well as normalized data. By looking at the table below, we can see that normalized data doesn't really improve the performance, and using less features will worsen the performance.

|               |       more features         || more features & normalized ||        less features       ||
|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
|               | test1 loss   | test2 loss   | test1 loss   | test2 loss   | test1 loss   | test2 loss   |
| Random Forest | 1737228.7359 | 6569618.5177 | 1750304.2386 | 6537426.7855 | 1787680.1939 | 6790996.1342 |

Test 2 loss is based on the median of rent from the training set, however, median rent does poorly on estimating the actual loss. \[Explain why?\]


## How to Improve Performance of the Model for Round 2
- Use external dataset, such as income or population around the neigborhood. In particular, we expect income to have meaningful influence on rent prediction. 
- Use the object fields (string) to train the models as well. They should be treated as categories and turned into integer.
- (Ensemble modeling) Combine different model predictions and average the prediction to get the final result.

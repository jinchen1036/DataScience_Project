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


A 200-300 word explanation of the expected performance of the model in terms of mean squared error and the key features driving the team’s modeling performance.


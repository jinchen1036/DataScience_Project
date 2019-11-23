## Strategies 
To find the best model with best performance, our approach was as followed:
1. Conduct EDA on the three given data sets (training set, test1 set, test2 set) by visualizing all the numerical fields that are not boolean (binary) values using histograms. We picked out the obvious outliers and see if they are truly outliers (i.e. if the value makes sense) by getting the min and max of the columns. We also identified the fields that contain null values.
2. Clean the data based on what we found during the EDA:
    - Combine the training data with test data set 1 and 2, and clean them together in the same way as they are from the same source and each field should have similar characteristics.
    - Use median value of each field to replace the outlier/null values. We used median instead of mean because outliers could have big influence on the mean.
    - Normalize the clean dataset so that the range of each field is from 0 to 1. This can make sure the influence of each field is equally distributed to the model.
3. Conduct statistic analysis to identify which fields to be used for training the model:
    - Use Ordinary Least Squares to see the relation of each field with the target (rent).
    - Use the fields with p-value equal to 0.
4. Try various models for regression problem and find the best one. As for hyperparamters, at first we used RandomizedSearchCV to randomize the hyperparameters for quick analysis purposes. Then we used GridSeachCV to find the best hyperparameters.
    - List of models that we have tried are:
        - LinearRegression
        - KNeighborsRegressor
        - DecisionTreeRegressor
        - GradientBoostingRegressor
        - AdaBoostRegressor (with based estimator as LinearRegression)
        - RandomForestRegressor
    - Train the models with training dataset, and calculate the loss of both test set 1 and 2:
        - Use the median of rent in training dataset as the true label for determining the test set 2 loss
        - Find the model with smallest loss for test set 1, and re-train that model with both training set and test set 1 to get the final model
        - Use the final model to predict the test set 2 rent
#### How to Improve Performance of the Model for Round 2
- Use external dataset(s), such as income or population around the neighborhood. In particular, we expect income to have meaningful influence on rent prediction. 
- Use the object fields (string) to train the models as well. They should be treated as categories and turned into integer.
- (Ensemble modeling) Combine different model predictions and average the prediction to get the final result.



## Model Performance
#### Comparison of All Model Results


This table contains the performance of various models we tried in terms of mean squared error. 
- Compared with using more features and less features.
    - more features = ['bedrooms', 'bathrooms', 'size_sqft', 'addr_zip', 'floor_count', 'min_to_subway', 'has_doorman', 
                                'has_elevator', 'has_dishwasher', 'is_furnished', 'has_gym', 'allows_pets', 'has_washer_dryer', 
                                'has_concierge', 'no_fee', 'has_pool', 'floornumber']
    - less features = ['bedrooms', 'bathrooms', 'year_built', 'addr_zip', 'bathrooms',' size_sqft', 'no_fee']
- Test 2 loss is based on the median of rent from the training set, however, median rent does poorly on estimating the actual loss.

|                   |       More Features        ||         Less Features      ||
|-------------------|--------------|--------------|--------------|--------------|
|                   | test1 loss   | test2 loss   | test1 loss   | test2 loss   |
| Linear Regression | 2870402.2234 | 5830874.4832 | 3130781.1625 | 5702636.3532 |
|   Decision Tree   | 2853766.8381 | 6596728.8758 | 2335347.4257 | 6989698.4956 |
|     KNeighbors    | 3911766.0539 | 3723764.7534 | 2305387.7376 | 6478483.8429 |
|    Ada Boosting   | 3059467.7381 | 4752455.2622 | 3172407.7560 | 4806204.5106 |
| Gradient Boosting | 4865627.4079 | 1133123.8333 | 4874866.8723 | 1160366.4162 |
|   Random Forest   | 1737228.7359 | 6569618.5177 | 1787680.1939 | 6790996.1342 |

Model Result Analysis
- Some models with less features have better results but not apply to all models
- KNN have biggest difference when using different number of features.  
- Gradient Boosting always have high loss in test dataset 1, but lowest in test dataset 2, so is using the median train dataset rent as the comparision of predict rent a bad idea? 
  - The model with lower test1 loss usually have higher test2 loss, again not think test2 loss is reliable without having the actual rent for test2
- The final best hyperparameters also shows that ensemble model using more estimator is not always good. 
- Random Forest have the lowest loss for test set 1, should be our final model.

#### Final Model - Random Forest
Analysis Random Forest performance compared with using more features, less features, as well as normalized data. By looking at the table below, we can see that normalized data doesn't really improve the performance, and using less features will worsen the performance.

|               |       more features        || more features & normalized ||        less features       ||
|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
|               | test1 loss   | test2 loss   | test1 loss   | test2 loss   | test1 loss   | test2 loss   |
| Random Forest | 1737228.7359 | 6569618.5177 | 1750304.2386 | 6537426.7855 | 1787680.1939 | 6790996.1342 |

Analysis
- Surprise that the normalization have not improve the model performance.
- Same model don't have same results for each training, could be the model focus on different features or the threshold value are changing. 


#### Analysis Median Train Dataset Rent with Test Set 1 actual and predict rent.
- Result -> The evaluation of the loss of the rent based the training set median rent cannot give us any useful information, see the table below

|                   |            Mean Square Error        |
|-------------------------------------------|-------------|
| Test1 actual rent VS median rent          | 7828536.681 | 
| Test1 predict rent VS median rent         | 5253709.207 |
| Test1 predict rent VS Test1 actual rent   | 1739729.277 |

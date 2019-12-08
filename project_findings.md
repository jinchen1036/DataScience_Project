(i) A markdown file entitled project_findings.md containing answers and supporting evidence for all of points in the Questions and Tasks section that follows.
(ii) A Jupyter notebook allowing for the complete replication of the modeling process.

## Questions and Tasks

#### 1. **Data Usage**
  * What outside data have you appended to the original data set? Why did you choose this data? 
    *  Outside data come from [incomebyzipcode.com](https://www.incomebyzipcode.com/newyork) through web scraping. 
        * [Extra Data](https://raw.githubusercontent.com/jinchen1036/DataScience_Project/master/ExternalDF/zipcode_income_withrentid.csv?token=AKASCZIDUHRZH2GD24ZKT4C55HFAU)
    *  Reason to Choose this data
        * We believe the income of the region can says a lot about the rent with below assumptions
            1. Rent should be in similar scale among the same region 
            2. Area with higher income will have higher rent compare to low income area
        * Limit of good and relevant dataset available
            1. Many dataset are not good match with the StreetEasy Dataset, as many datasets are with bin, bbl or zipcode information.
            2. Sample Datasets Found  
                * [DOF: Cooperative Comparable Rental Income Dataset](https://data.cityofnewyork.us/City-Government/DOF-Cooperative-Comparable-Rental-Income-Citywide-/myei-c3fa)
                    * Only match about 30% data with bbl, and as bbl are duplicated in both datasets so this match cannot be reliable 
                * [Zip Codes and Stats](https://www.kaggle.com/jakerohrer/zip-codes-and-stats)
                    * Median Household Income [2006-2010], outdated data compare to the data we using. 
                    * Bad idea using outdated data since it will mislead our model. 
  * Does the inclusion of this additional data raise any ethical considerations?
    *   No, as the additional data is the median income based on zip code which actually come from Census data. We did not reveal any personal information. 

#### 2. **Data Exploration**
  * What outliers present issues for your analysis? How have you chosen to handle them? Why?
    * Outliers we found that seems to be incorrect - [Reference](https://github.com/jinchen1036/DataScience_Project/blob/master/Practice_Analysis_Part1/FeatureAnalysis.ipynb)
        1. bathroom
            * Most of bathroom number are between 1 to 5
            * 2 places with 12 bathrooms but with rent 2700 and 3200, 1 places with 20 bathrooms with rent 5000. Also 8 places with 0 bathroom
            * Based on intuitive, these outliers are wrong data, as the rents are too low considering their bathroom numbers and there shouldn't be places that are without bathroom.
        2. size_sqft
            * Most of size_sqft are between hundreds to 4000
            * 3 places with size_sqft over 8500 with rent under 4000. Comparing to places with size_sqft of around 6000 that charge 20K, we can tell these 3 places have wrong size_sqft. 
        3. year_built
            * Most of year_built are between 1800 to 2019
            * 91 places with year_built of 0, they must be missing values.
        4. min_to_subway
            * Most of min_to_subway are within 100
            * 7 places with min_to_subway exactly as `103343.6167`. By looking into their zip codes, we found they locate in Brooklyn with many subway stations near by, these must be wrong data.
    * Handle Outliers
        * We treat these outliers the same way as missing data, since we consider these outliers wrong data so we will replace them with new values.
        * We use the median value of that column to replace its value.
            
  * To what extent do missing values pose a challenge for your analysis? How have you chosen to handle them? Why?
    * Missing values always create some challenge since we cannot get the correct information about each building and have to replace it with other reasonable value. Even the replaced value is tend to be close to the actual value, but still is not exactly correct and will influence on the final result from the model. 
    * We used median value of the associate column to replace all the missing values, since we cannot train the model with any missing value.
    * Reason to pick the median value is because mean value can be influence by outliers which might be bias to represent the column. However, median value will not be influence by the outliers and is will not have big effect to the model's decision on the 
  * Are there any other aspects of the data your exploration shows might be problematic?
    * There are some problematic data.
        * 91 places with year_built as 0.
        * 8 places with 0 bathroom.
        * Over thousands of places with 0 size_sqft.
        * 7 places with min_to_subway exactly as `103343.6167`
    * As these problematic data are either `0` or exactly as `103343.6167`, one question that raised is that is the system automatic fill some data when some fields of the building record when it is missing, since both `size_sqft` and `bathroom` fields not missing any data.
     
  * Create at least one visualization that demonstrates the predictive power of your data.
  ![vis_<2000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_lessthan2000.png?raw=true=400x300)
  ![vis_2000-3000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_2000between3000.png?raw=true)
  ![vis_3000-4000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_3000between4000.png?raw=true)
  ![vis_4000-5000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_4000between5000.png?raw=true)
  ![vis_5000-6000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_5000between6000.png?raw=true)
  ![vis_6000-10000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_6000between10000.png?raw=true)
  ![vis_10000-50000](https://github.com/jinchen1036/DataScience_Project/blob/master/Visualization/RF_Prediction_rent_10000between50000.png?raw=true)
    
#### 3. **Transformation and Modeling**
  * Describe 5-10 features you think play the biggest role in your model. 
    * How did you create these features?
       * Pick three object feature to use -  addr_city, neighborhood, and borough.
       * Method: We used a fillna method to make sure there is no null value in these three columns. Then we used a LabelEncoder method to change their datatype from object to numerical so they can use in model training. 
    
    * How do you know these features are playing key roles?
    1. Income - By zipcode
        - From external source, get from web scrapping
        - Based on the zipcode income, we can get a rough idea of how much people are willing or can pay for their rent, this will narrow down the prediction range of the rent.
    2. Neighborhood 
        - Convert the string to integer label, so can be used for model training
        - Model can used that to group the house by neighborhood, and most likely the rent will be very similar within the same neighborhood, as neighborhood are small region than zipcode
    3. bathrooms
        - We remove the incorrect outliers and greatly decrease the range of the number of bathrooms
        - Smaller range help the decision tree estimators separate the data, as without the big influence of large and wrong data
        - As for rent, people are more likely to have more bathroom available as people are less likely to share the bathroom with ours, especially for the stranger roommates. Demand increase the pricing, also more construction are need for more bathroom. Thus, number of the bathroom become a key for determine the rent, and usually more bathroom means bigger place then cost more rent.
    4. floornumber
        - We used the original data for the floornumber, as it's data seems reasonable. 
        - But it does have a impact in our model training, as most of house are between 0 to 8, but it is a great distribution of house in floor number above 8, which decision tree can do detail/better prediction from these house, as it can separate the feature better.
    5. size_sqrt
        - We thought this might be the biggest factor since the price of a property is directly related to its area.
        - We just used the original numbers from the dataset.
    * We see the different test loss which one of the features are missing, even random forest don't have stabilize loss because of its randomness, but we can still see the same difference by running the same model with same features multiple times and compare the mean.
    * Also these features on the OLS Regression Analysis all with p-value of 0, which again proof the usefulness of these features.
  * Describe how you are implementing your model. Why do you think this works well?
    1. From GridSeachCV, we get our best hyperparameter for RandomForest as `max_features=10,n_estimators=320,bootstrap=True`
        - The max_features is the highest number we tried nor the n_estimators, which proof that using more features/estimators are always better and simple model sometimes are better.
        - Also with `bootstrap=True`, introduce more randomness to each estimator in the random forest, which allow the final prediction be more generalize fit for any type of data.
    2. Key point of random forest is that each estimator is independent from each other, so even if some model don't do well, but majority of it still predict reasonable rent.
    3. Another key point is we do a great job in feature extraction, which help model to analysis the best feature for rent prediction 
  * Describe your methodology for selecting your model. Why do you think this type of model works well?
    1. Try various models for regression problem and find the best one. 
       - Used GridSeachCV to find the best hyperparameters.
       - List of models that we have tried are:
            - LinearRegression
            - KNeighborsRegressor
            - DecisionTreeRegressor
            - GradientBoostingRegressor
            - AdaBoostRegressor (with based estimator as LinearRegression)
            - RandomForestRegressor
    2. Train the models with training dataset, and calculate the loss of both test set 1 and 2:
        - Mainly focus on test 1 loss, as we don't know the really rent for test 2
        - Find the model with smallest loss for test set 1
    3. Random Forest is combine of many uncorrelated tree model to get the final prediction
        - For a single decision tree model, we already get a loss of `2853766.8381` for test 1 (see [ModelTrainResult](https://github.com/jinchen1036/DataScience_Project/blob/master/Model_Result_Part1/ModelTrainResult2.ipynb))
        - Using multiple uncorrelated models will outperform the individual model, as using these models produce ensemble predictions which are more accurate.
        - Last, in compare to all other individual model performance, DecisionTreeRegressor is the best one, thus applying ensemble method with decision tree will be even better.

#### 4. **Metrics, Validation, and Evaluation**
  * How well do you think you model will perform on the hold out test set? How do you know? 
    - By analysis the loss of the test 1, we can get a rough idea of the model performance on test 2 and 3, as they are from same data source and not big period apart from each other. 
    - We give out the comparision of median rent of train dataset with test 2 prediction dataset
        - Gradient Boosting always have high loss in test dataset 1, but lowest in test dataset 2
        - The model with lower test1 loss usually have higher test2 loss
        - Also after submission of test 2 prediction, we conclude our thought which are compare median rent to predict rent is very unreliable, but looking at test 1 loss of the model is more reliable. 
  * Is your model useful? Why or why not?
     - It is quite useful to predict some common apartment rents. From the visualizations of the predictive power of our data, we can see that for most of the apartments with common rents range from $1500 to $4000 (not ridiculously luxury apartments), it performs pretty well that our predictive rents are pretty close to the actual rents. But for some cases, like some luxury apartments with rents range from $10000 to $50000, our predictions are not so good.
  * Are there any special cases in which your model works particularly well or particularly poorly? 
     - Based on the visualizations of the predictive power, within the price range of $1500 to $4000, our model has pretty good estimations. It's more accurate when the prices are lower, and less accurate when the prices are higher.
  * Create at least one visualization that demonstrates the predictive power of your model.
    - [See 'rf_predictions.png' in main directory](https://raw.githubusercontent.com/jinchen1036/DataScience_Project/master/rf_predictions.png?token=AJXDIGHAKAJJA6CTSMW3ODS56WVRK)

#### 5. **Conclusion**
  * How would you use this model?
     - Listing sites could use this model to give landlords a suggestion rent for their apartments.
     - People looking for apartments could use this to estimate whether an apartment is within a reasonable price range.
  * If you could have additional modeling features, what would they be? 
     - Through analysis of our dataset, we can say that not all the data in the dataset are correct. Such as the outliers we remove from some of the features.
     - Then if we can get a confidence level of the data correctness for each row/house, then we can identify some of the extreme case, which in most time is because of the wrong data it give. 
     - Also, it will be make more sense if the model are train separately to get the range of rent rather than the actual rent, since there are always some deviation among the prediction and actual rent. However, we achieve better accuracy if provide the range of the rent, but the range should be within $100 difference not large scale. 
  * Would you rather have more data, or more features?
     - No doubt - More DATA!
     - More features are not always better, which can be proof that we only use 10 as our maximum features for each estimator in random forest, even when we actually have more features.
     - More importantly, more data means the model can learn more generalize feature/characterize of the rent, which allow it to fit for various test dataset. 
     - Another big issue is with increase number of features in trainning set, then will also require more features in test set, and in really life it is hard to accomplish. However, it will be easier to collect rough features but more data/samples.


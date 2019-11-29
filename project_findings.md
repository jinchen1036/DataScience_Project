(i) A markdown file entitled project_findings.md containing answers and supporting evidence for all of points in the Questions and Tasks section that follows.
(ii) A Jupyter notebook allowing for the complete replication of the modeling process.

## Questions and Tasks

#### 1. **Data Usage**
  * What outside data have you appended to the original data set? Why did you choose this data? 
    *  Outside data come from [incomebyzipcode.com](https://www.incomebyzipcode.com/newyork) through web scraping. 
        * [Extract Data](https://raw.githubusercontent.com/jinchen1036/DataScience_Project/master/ExternalDF/zipcode_income_withrentid.csv?token=AKASCZIDUHRZH2GD24ZKT4C55HFAU)
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
                    * Bad idea using outdated data since it will misleading our model. 
  * Does the inclusion of this additional data raise any ethical considerations?
    *   No, as the additional data is the median income based on zip code which actually come from Census data. We did not reveal any personal information. 

#### 2. **Data Exploration**
  * What outliers present issues for your analysis? How have you chosen to handle them? Why?
    * Outliers we found that seems like incorrect - [Reference](https://github.com/jinchen1036/DataScience_Project/blob/master/Practice_Analysis_Part1/FeatureAnalysis.ipynb)
        1. bathroom
            * Most of bathroom number are between 1 to 5
            * 2 places with 12 bathrooms but with rent 2700 and 3200, 1 places with 20 bathrooms with rent 5000. Also 8 places with 0 bathroom
            * Based on intuitive, these outliers are wrong data, as the rent are too long compare its bathroom number and should have place that are without bathroom.
        2. size_sqft
            * Most of size_sqft are between hundreds to 4000
            * 3 places with size_sqft over 8500 with rent under 4000, by compare to the place with size_sqft around 6000 have rent over 20K, we can decide these 3 places have wrong size_sqft. 
        3. year_built
            * Most of year_built are between 1800s to 2019
            * 91 places with year_built as 0, without any comparision we know these are wrong.
        4. min_to_subway
            * Most of min_to_subway are within 100
            * 7 places with min_to_subway exactly as `103343.6167`, without investigation we know these are wrong data. Also by look into their zip code which are around Brooklyn with many subways around it, so again it is wrong data.
    * Handle Outliers
        * We treat these outliers same way as missing data, since we consider these outliers are wrong so we have to replace it with new values.
        * We use the median value of its associate column to replace its value.
            
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

#### 3. **Transformation and Modeling**
  * Describe 5-10 features you think play the biggest role in your model. 
    * How did you create these features?
    * How do you know these features are playing key roles?
  * Describe how you are implementing your model. Why do you think this works well?
  * Describe your methodology for selecting your model. Why do you think this type of model works well?

#### 4. **Metrics, Validation, and Evaluation**
  * How well do you think you model will perform on the hold out test set? How do you know? (b) Is your model useful? Why or why not?
  * Are there any special cases in which your model works particularly well or particularly poorly? (d) Create at least one visualization that demonstrates the predictive power of your model.

#### 5. **Conclusion**
  * How would you use this model?
  * If you could have additional modeling features, what would they be? 
  * Would you rather have more data, or more features?
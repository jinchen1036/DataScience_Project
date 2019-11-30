import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# When display in pycharm
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)

def load_dataset(numeric=True, extra_dataset=False):
    '''
    :param numeric: if True will return only numeric columns
    :param extra_dataset: if True will return extract income information by zipcode
    :return combined dataset(dataframe), train_df_index(series), test_set1_index(series), test_set3_index(series)
    '''
    train_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
    test_set1_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
    test_set2_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test2.csv', index_col=0)
    test_set3_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test3.csv', index_col=0)

    train_df_index = train_df.index
    test_set1_index = test_set1_df.index
    test_set2_index = test_set2_df.index
    test_set3_index = test_set3_df.index
    combine_df = pd.concat([train_df,test_set1_df,test_set2_df,test_set3_df])
    combine_df['addr_zip'] = combine_df['addr_zip'].astype(np.int64)

    if numeric:
        combine_df = combine_df.select_dtypes(include=[np.number])
    if not extra_dataset:
        combine_df.drop(['bin', 'bbl', 'building_id'], axis=1, inplace=True)
    else:
        external_df= pd.read_csv('https://raw.githubusercontent.com/jinchen1036/DataScience_Project/master/ExternalDF/zipcode_income_withrentid.csv?token=AKASCZIDUHRZH2GD24ZKT4C55HFAU', index_col=0)
        combine_df['income'] = external_df.income

    return combine_df, train_df_index, test_set1_index,test_set2_index, test_set3_index


def clean_data(dataset):
    '''
    Clean the dataset, deal with missing value, outliers and wrong data.
    Data Analysis Result based on FeatureAnalysis
       bathrooms -> replace any value greater than 7 or equal to 0 with median
       size_sqft -> replace any value greater than 8000 or equal to 0 with median
       year_built -> replace any value equal to 0 with median
       min_to_subway -> replace any value greater than to 100 with median
       median_income_11249 -> 83,407
       fill_NA -> with median
    :param dataset(dataframe): the dataset to be clean
    :return clean dataset (dataframe)
    '''

    # Get all column with NAN and fill with median value
    dataset.income.fillna(83407,inplace = True)
    dataset = dataset.select_dtypes(include=[np.number])
    must_process_columns = ['bathrooms','size_sqft','year_built','min_to_subway']
    nan_columns = dataset.columns[dataset.isna().any()].tolist()
    nan_columns = list(set(nan_columns + must_process_columns))
    for column in nan_columns:
        if column == 'rent':  # not make any change to rent
            continue
        median_value = (dataset[column]).median()
        dataset[column].fillna(median_value,inplace = True)
        if column == 'bathrooms':
            dataset.loc[(dataset['bathrooms'] == 0) | (dataset['bathrooms'] > 7), 'bathrooms'] = median_value
        elif column == 'size_sqft':
            dataset.loc[(dataset.size_sqft == 0) | (dataset.size_sqft > 8000), 'size_sqft'] = median_value
        elif column == 'year_built':
            dataset.loc[dataset.year_built == 0, 'year_built'] = median_value
        elif column == 'min_to_subway':
            dataset.loc[dataset.min_to_subway > 100, 'min_to_subway'] = median_value
    return dataset

def normalized_dataset(dataset):
    '''
    Normalize all columns with scale 0 to 1, return columns with all type as float64
    :param dataset(dataframe): the dataset to be normalize
    :return normalized dataset (dataframe)

    '''
    for column in list(dataset.columns):
        if column == 'rent':  # not make any change to rent
            continue
        min = dataset[column].min()
        max = dataset[column].max()

        dataset[column] = dataset[column].astype(np.float64)
        if min == 0 and max == 1:  # not need to fix boolean values
            continue
        dataset[column] = (dataset[column] - min) / (max - min)
    return dataset


def transform_data(dataset):
    '''
    Convert 'addr_city','neighborhood','borough' from string the int
    :param dataset(dataframe): the dataset to be process
    :return processed dataset (dataframe)

    '''
    le = LabelEncoder()
    process_columns = ['addr_city','neighborhood','borough']
    for column in process_columns:
        dataset[column].fillna("NA",inplace = True)
        dataset[column]=le.fit_transform(dataset[column])
    return dataset

# combine_df, train_df_index, test_set1_index, test_set2_index ,test_set3_index= load_dataset(numeric=False, extra_dataset=True)
# combine_df = transform_data(combine_df)
# combine_df = clean_data(combine_df)

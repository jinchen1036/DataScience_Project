import re
import pandas as pd
import numpy as np
from sodapy import Socrata
from Helper.StaticParameters import Parameters

def get_original_bbl():
    train_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
    test_set1_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
    test_set2_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test2.csv', index_col=0)
    test_set3_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test3.csv', index_col=0)
    combine_df = pd.concat([train_df['bbl'], test_set1_df['bbl'], test_set2_df['bbl'],test_set3_df['bbl']])
    # del train_df,test_set1_df,test_set2_df, test_set3_df
    return combine_df


def get_external_income_df():
    client = Socrata("data.cityofnewyork.us", None)
    results = client.get("myei-c3fa", limit=32000)
    results_df = pd.DataFrame.from_records(results)

    all_external_df = {}
    for i in range(4):
        if i > 0:
            feature_columns = [feature + '_%d'%i for feature in Parameters.rental_income_columns]
        else:
            feature_columns = Parameters.rental_income_columns

        print(feature_columns)
        all_external_df['df_%d'%i] = results_df.loc[:,feature_columns]
        all_external_df['df_%d' % i] = all_external_df['df_%d'%i].dropna(axis=0)
        for feature in feature_columns:
            if feature.startswith('boro_block'):
                all_external_df['df_%d' % i]['bbl'] = all_external_df['df_%d'%i][feature].str.replace(r'-', '').astype(np.int64)
                all_external_df['df_%d' % i].drop([feature], axis=1, inplace=True)
            elif not feature.startswith('neighborhood'):
                all_external_df['df_%d' % i][feature] = all_external_df['df_%d' % i][feature].astype(np.float64)
            if bool(re.search(r'\d', feature)):
                cut_index = feature.rindex('_')
                all_external_df['df_%d' % i].rename(columns={feature: feature[:cut_index]}, inplace=True)
    combine_income_df = pd.concat([all_external_df['df_0'],all_external_df['df_1'],all_external_df['df_2'],all_external_df['df_3']])
    return combine_income_df



combine_df = get_original_bbl()
combine_df = combine_df.to_frame()
combine_income_df = get_external_income_df()
unique_rent_bbl = combine_df.drop_duplicates('bbl')
unique_income_bbl=combine_income_df.drop_duplicates('bbl')

# combine_dfs = combine_df.merge(combine_income_df,on='bbl',how='left')
# combine_df.to_csv("external_income.csv", header=True)



new_df=pd.merge(unique_rent_bbl, unique_income_bbl, on='bbl', how='left')
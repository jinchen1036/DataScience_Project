import requests
import pandas as pd


def income_extractor(zip_code_list):
    incomes = []
    income_dict = {}
    i = 0
    for zip_code in zip_code_list:
        if zip_code in income_dict.keys():
            incomeNum = income_dict[zip_code]
        else:
            r = requests.get(url="https://www.incomebyzipcode.com/newyork/%d"%zip_code)
            html = r.content.decode('utf-8')
            try:
                income = html[html.index('hilite'):html.index('hilite') + 20]
                income = income[income.index("$") + 1: income.index("<")]
                incomeNum = float(income.replace(',', ''))
            except:
                print("Not found for %d" % zip_code)
                incomeNum = None
        incomes.append(incomeNum)
        income_dict[zip_code] = incomeNum
        i+=1
        if i %100 == 0:
            dict_to_csv(income_dict)
    return incomes

def get_original_addr_zip():
    train_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_train.csv', index_col=0)
    test_set1_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
    test_set3_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test3.csv', index_col=0)
    combine_df = pd.concat([train_df['addr_zip'], test_set1_df['addr_zip'],test_set3_df['addr_zip']])
    # del train_df,test_set1_df,test_set2_df, test_set3_df
    return combine_df.to_frame()

def dict_to_csv(dict):
    with open('zipcode_income.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s,%s\n" % (key, dict[key]))

combine_df = get_original_addr_zip()
combine_df['income'] = income_extractor(combine_df['addr_zip'])
combine_df.to_csv("zipcode_income_withrentid.csv", header=True)
# test_set1_df = pd.read_csv('https://grantmlong.com/data/SE_rents2018_test1.csv', index_col=0)
# test_df= test_set1_df['addr_zip'][:10].to_frame()
# test_df['income'] = income_extractor(test_df['addr_zip'])
# test_df.to_csv("zipcode_income_withrentid.csv", header=True)
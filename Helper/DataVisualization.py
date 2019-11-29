import statsmodels.api as sm

def statistic_analysis(dataset):
    '''
    Estimating the unknown parameters (all columns in the dataset) influence to the rent in a linear regression model
    :param dataset: dataset to be analysis
    :return: void function, will print the OLS analysis summary
    '''
    fixed_set = dataset.dropna(axis=0)
    fixed_set = sm.add_constant(fixed_set)
    columns = list(fixed_set.columns)
    columns.remove('rent')

    est = sm.OLS(fixed_set['rent'],
                 fixed_set[columns].astype(float)).fit()
    print(est.summary())
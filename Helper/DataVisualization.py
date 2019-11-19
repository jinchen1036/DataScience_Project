import statsmodels.api as sm

def statistic_analysis(dataset):
    fixed_set = dataset.dropna(axis=0)
    fixed_set = sm.add_constant(fixed_set)
    columns = list(fixed_set.columns)
    columns.remove('rent')

    est = sm.OLS(fixed_set['rent'],
                 fixed_set[columns].astype(float)).fit()
    print(est.summary())
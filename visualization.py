import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Actual Rent
test1_df = pd.read_csv('Predict_CSVFiles_Part1/SE_rents2018_test1.csv')
test1_df = test1_df[['rent']]

# Rent predicted by random forest trained with raw data
test1_pred_df = pd.read_csv('Predict_CSVFiles_Part1/RandomForest_Prediction_TestSet1.csv')
test1_df['pred_rent'] = test1_pred_df[['predictions']]

# Rent predicted by random forest trained with normalized data
test1_pred_norm_df = pd.read_csv('Predict_CSVFiles_Part1/RandomForest_Prediction_TestSet1_Normalize.csv')
test1_df['pred_norm_rent'] = test1_pred_norm_df[['predictions']]

breaks = [2000,3000,4000,5000,6000,10000,50000]
for i in range(len(breaks)):
    fig_name = "%d"%breaks[i]
    if i == 0:
        fig_name = "lessthan%s"%fig_name
        sampleSet = test1_df.loc[(test1_df['rent'] <= breaks[i])]
    else:
        sampleSet = test1_df.loc[(breaks[i-1]< test1_df['rent'])&(test1_df['rent'] <= breaks[i])]
        fig_name = "%dbetween%s" % (breaks[i-1],fig_name)

    sampleSet['x'] = np.arange(sampleSet.shape[0])
    ax = sampleSet.plot(x="x", y="rent", kind="scatter", label='rent', color="C2")
    sampleSet.plot(x="x", y="pred_rent", kind="scatter", ax=ax, label='predict rent', color="C3")
    sampleSet.plot(x="x", y="pred_norm_rent", kind="scatter", ax=ax, label='pred_norm_rent', color="C1")
    plt.title('RF_Prediction_rent_%s.png'%fig_name)
    plt.legend()
    plt.savefig('Visualization/RF_Prediction_rent_%s.png'%fig_name)


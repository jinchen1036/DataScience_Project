from Helper.DataLoader import *
from Helper.SearchCVModels import CV_Model
from sklearn.metrics import mean_squared_error


class ModelProcessor:

    def train_model(self, classifier, parameters,feature_columns, train_df, test_df, grid = False, train_target='rent',cv_split = 6, trainWithTest1 = True, test2_df=None):
        SearchCV = CV_Model(GridSearch=grid)
        SearchCV.train_model(classifier, parameters, train_df[feature_columns],train_df[train_target], cv_split = cv_split)
        SearchCV.train_best_model(train_df[feature_columns],train_df[train_target]) # train model with all train data again

        self.test_set = test_df.copy()
        self.test_set['predictions'] = SearchCV.pred_target(test_df[feature_columns])

        if trainWithTest1:
            loss = mean_squared_error(self.test_set['predictions'], self.get_fake_rent(train_df[train_target], test_df.shape[0]))
            print("Test Set 2 - Loss: %.4f" % loss)

        else:
            loss = mean_squared_error(self.test_set['predictions'],test_df[train_target])
            print("Test Set 1 - Loss: %.4f" % loss)

            test2_pred = SearchCV.pred_target(test2_df[feature_columns])
            loss = mean_squared_error(test2_pred, self.get_fake_rent(train_df[train_target], test2_df.shape[0]))
            print("Test Set 2 - Loss: %.4f" % loss)

    def get_fake_rent(self, train_target, test_df_shape):
        return np.ones(test_df_shape) * train_target.median()

    def generate_result_csv(self, file_name):
        self.test_set['predictions'].to_csv(file_name, header=True)
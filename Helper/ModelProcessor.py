from Helper.DataLoader import *
from SearchModel_Part1.SearchCVModels import CV_Model
from sklearn.metrics import mean_squared_error


class ModelProcessor:
    
    def train_model(self, classifier, parameters,feature_columns, train_df, test_df, grid = False, train_target='rent',cv_split = 6, trainWithTest1 = True, test2_df=None):
        '''
        :param classifier: Model to be used to predict
        :param parameters: Dict of hypeparameter to be test by searchCV
        :param feature_columns: list of features to be used for training the model
        :param train_df: Training dataset
        :param test_df: Test dataset
        :param grid: if True using GridSearchCV else using RandomizedSearchCV
        :param train_target: string, target column
        :param cv_split: int, number of split for cross validation
        :param trainWithTest1: if True then train_df must contain both train and test1 data, test_df will be the final test set
                               if False must provide test2_df and will test the model performance on test2_df
        :param test2_df: Test dataset 2
        :return:
        '''
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
        '''
        :param train_target: the labels of the training dataset
        :param test_df_shape: number of sample in the test dataset
        :return: array with size test_df_shape, all with value as training label median value
        '''
        return np.ones(test_df_shape) * train_target.median()

    def generate_result_csv(self, file_name):
        '''
        Save the test set prediction to csv file
        :param file_name: the name of the csv file to be save
        :return: void function
        '''
        self.test_set['predictions'].to_csv(file_name, header=True)

    def train_final_model(self,classifier, train_df, test_df,feature_columns,train_target='rent',trainWithTest1 = True, test2_df=None):
        '''
        :param classifier: Model to be used to predict the final rent
        :param train_df: Training dataset
        :param test_df: Test dataset
        :param feature_columns: list of features to be used for training the model
        :param train_target: string, target column
        :param trainWithTest1: if True then train_df must contain both train and test1 data, test_df will be the final test set
                               if False must provide test2_df and will test the model performance on test2_df
        :param test2_df: Test dataset 2
        :return: void, will print both train and test loss, as well as set ModelProcessor.test_pred
        '''
        self.classifier = classifier
        self.classifier.fit(train_df[feature_columns], train_df[train_target])

        self.train_pred = self.classifier.predict(train_df[feature_columns])
        loss = mean_squared_error(self.train_pred, train_df[train_target])
        print("Train Loss: %.4f" % loss)

        self.test_pred = self.classifier.predict(test_df[feature_columns])
        if trainWithTest1:
            loss = mean_squared_error(self.test_pred, self.get_fake_rent(train_df[train_target], test_df.shape[0]))
            print("Test Set Loss: %.4f" % loss)
            return self.train_pred, self.test_pred
        else:
            loss = mean_squared_error(self.test_pred, test_df[train_target])
            print("Test Set 1 Loss: %.4f" % loss)
            self.test2_pred = self.classifier.predict(test2_df[feature_columns])
            t2_loss = mean_squared_error(self.test2_pred, self.get_fake_rent(train_df[train_target], test2_df.shape[0]))
            print("Test Set 2 Loss: %.4f" % t2_loss)
            return self.train_pred, self.test_pred, self.test2_pred



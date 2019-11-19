from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error,make_scorer

class CV_Model():
    def __init__(self,GridSearch=True):
        if GridSearch:
            self.searchCV = GridSearchCV
        else:
            self.searchCV = RandomizedSearchCV
        self.target = 'rent'

    def train_model(self,estimator, param_grid, train_data, feature_columns, cv_split = 6):
        score = make_scorer(mean_squared_error, greater_is_better=False)
        # para_grid = {'knn__n_neighbors': np.arange(1, 50)}
        self.model_cv = self.searchCV(estimator, param_grid, cv=cv_split, scoring=score)
        self.model_cv.fit(train_data[feature_columns], train_data[self.target])
        print('Best Estimator',self.model_cv.best_estimator_)
        print('Best Parameters: ',self.model_cv.best_params_)
        print('Best Score: ',self.model_cv.best_score_)

    def test_model(self, test_data,feature_columns):
        self.best_model = self.model_cv.best_estimator_
        y_predicted = self.best_model.predict(test_data[feature_columns])
        loss = mean_squared_error(test_data[self.target], y_predicted)
        print("Test Loss: %.4f" % loss)
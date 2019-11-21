from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error,make_scorer

class CV_Model():
    def __init__(self,GridSearch=True):
        if GridSearch:
            self.searchCV = GridSearchCV
        else:
            self.searchCV = RandomizedSearchCV
        self.target = 'rent'

    def train_model(self,estimator, param_grid, train_data, train_target, cv_split = 6):
        score = make_scorer(mean_squared_error, greater_is_better = False)
        self.model_cv = self.searchCV(estimator, param_grid, cv=cv_split, scoring=score)
        self.model_cv.fit(train_data, train_target)
        print('Best Estimator',self.model_cv.best_estimator_)
        print('Best Parameters: ',self.model_cv.best_params_)
        print('Best Score: ',self.model_cv.best_score_)

    def test_model(self, test_data, test_target):
        y_predicted = self.pred_target(test_data)
        loss = mean_squared_error(test_target, y_predicted)
        print("Test Loss: %.4f" % loss)
        return y_predicted

    def pred_target(self, test_data):
        self.best_model = self.model_cv.best_estimator_
        y_predicted = self.best_model.predict(test_data)
        return y_predicted
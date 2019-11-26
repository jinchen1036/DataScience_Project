import numpy as np
class Parameters:
    def __init__(self):
        self.feature_columns = ['bedrooms','bathrooms','size_sqft','addr_zip','floor_count','min_to_subway','has_doorman',
                                'has_elevator','has_dishwasher','is_furnished','has_gym','allows_pets','has_washer_dryer',
                                'has_concierge','no_fee','has_pool','floornumber']
        self.dtree_parameter = {
            'max_depth': np.arange(5, 55, 5),
            'min_samples_split': np.arange(10, 200, 10),
            'max_features': np.arange(4, 17, 2),
        }
        self.dtree_less_parameter = {
            'max_depth': np.arange(5, 55, 5),
            'min_samples_split': np.arange(10, 200, 10),
            'max_features': np.arange(4, 10, 2),
        }
        self.knn_parameter ={
            'n_neighbors': np.arange(2, 22, 2)
        }
        self.gradient_parameter = {'n_estimators': np.arange(100, 500, 50),'min_samples_split': np.arange(10, 200, 10),
          'learning_rate': [0.001, 0.0001, 0.00001]}

        self.ada_parameter = {'n_estimators': np.arange(100, 500, 50),'learning_rate': [0.001, 0.0001, 0.00001]}

        self.linreg_parameter = {'fit_intercept': [True, False] }


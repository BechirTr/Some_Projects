import numpy as np

# model libraries import
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted



class Model_reg(BaseEstimator, RegressorMixin):
  '''
    These Model is the aggregation of different models. 
    xgboost, lightgbm and catboost Regressor.
    The args where fixed before hand by us.
  '''
  def __init__(self, demo_param='demo'):
    self.demo_param = demo_param
  
  def fit(self, X, y):
    # Check that X and y have correct shape
    X, y = check_X_y(X, y)
    # we work on the log of y 
    y = np.log(y)
    # Intiliase the models 
    self.gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1,
                                   max_depth=6, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    
    self.lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=10000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.5,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    
    args = { 'random_state': 42, 'n_estimators': 5000,
            'max_depth': 6, 'learning_rate': 0.01,
            'colsample_bytree': 0.99, 'tree_method':"gpu_hist"}
    self.xgb = xgb.XGBRegressor(objective ='reg:squarederror', 
                  n_jobs = -1,**args)

    # split data because we have 2 training steps
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size= 0.15)
    
    # Fit the models
    self.gb.fit(X_train, y_train)
    print("gb done")
    self.lgb.fit(X_train, y_train)
    print("lgb done")
    self.xgb.fit(X_train, y_train)
    print("xgb done")

    # make predictions to tune the aggragating parameters
    y_pred1 = self.gb.predict(X_test)
    y_pred2 = self.lgb.predict(X_test)
    y_pred3 = self.xgb.predict(X_test)

    # Learn the aggregating coefficients for model
    param = (0,0,0) 
    mae_min = 99
    y_pred = list()
    for a in np.arange(0.01,0.99,0.01):
        for b in np.arange(0.01,0.99,0.01):
            if a+b < 1:
                y_pred = a*y_pred1 + b*y_pred2 + (1-a-b)*y_pred3
                mae = mean_absolute_error(y_test, y_pred)
                if mae < mae_min :
                    mae_min = mae
                    param = (a,b,1-a-b)

    # save the best aggregating coefficient 
    self.abc = param
    # Return the regressor
    return self

  def predict(self, X):
    # Check is fit had been called
    if self.fit == False:
        check_is_fitted(self.gb)
    # Input validation
    X = check_array(X)
    
    # compute the prediction using diffrent parameters
    y_pred1 = self.gb.predict(X)
    y_pred2 = self.lgb.predict(X)
    y_pred3 = self.xgb.predict(X)
    
    # compute the final prediction
    predictions =  self.abc[0]*y_pred1 + self.abc[1]*y_pred2 + self.abc[2] * y_pred3
    # apply the exp on the predictions to nulify the effect of the log applied previously
    predictions = np.exp(predictions)
    # return predictions
    return predictions
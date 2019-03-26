import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import sklearn
import data_preprocess

train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
test_x = data_preprocess.test_x
test_y = data_preprocess.test_y

parameters = {'n_estimators':[10,20,30,50],
              'max_depth':[3,4,5,6,7,8,9,10,20],
              'max_features': range(20, 26)}
model = ensemble.RandomForestClassifier(random_state=20)
# grid = model_selection.GridSearchCV(model, cv=10, param_grid=parameters,scoring='neg_mean_squared_error')
# grid.fit(train_x, train_y)
# print(grid.best_params_)
# print(grid.best_score_)
# result = grid.best_estimator_.predict(test_x)
# mse = metrics.mean_squared_error(result, test_y)
# print(mse)
model = ensemble.RandomForestClassifier(max_depth=20, max_features=22, n_estimators=50, random_state=20)
model.fit(train_x, train_y)

test_abbr = data_preprocess.test_abbr
test_abbr.dropna(axis=1,inplace=True)
test_abbr.drop(['survey_time'],axis=1,inplace=True)
result = pd.DataFrame(model.predict(test_abbr),
                      index= test_abbr['id'],columns=["happiness"])
result.to_csv('1.csv')




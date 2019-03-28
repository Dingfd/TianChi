import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
import sklearn
from matplotlib import pyplot as plt
import data_preprocess

train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
test_x = data_preprocess.test_x
test_y = data_preprocess.test_y

# parameters = {'max_features':range(18,26)
#               }
# model = ensemble.RandomForestClassifier(max_depth=18,n_estimators=53,random_state=20)
# grid = model_selection.GridSearchCV(model, cv=10, param_grid=parameters,scoring='neg_mean_squared_error')
# grid.fit(train_x, train_y)
# print(grid.best_params_)
# print(grid.best_score_)
# result = grid.best_estimator_.predict(test_x)
# mse = metrics.mean_squared_error(result, test_y)
# print(mse)
#1RF maxdep20, maxfeature22, n_estimator50,random_state20  score 0.894
#2RF maxdep20, maxfeature22, n_estimator64,63,random_state20  score 0.879
#3RF maxdep20, maxfeature22, n_estimator53,random_state20  score 0.881    cross validation
#4RF maxdep18, maxfeature22, n_estimator53,random_state20, score 0.8905   cross validation -0.72933

model = ensemble.RandomForestClassifier(max_depth=18, max_features=22, n_estimators=53, random_state=20)

# model.fit(train_x, train_y)
# result = model.predict(test_x)
# mse = metrics.mean_squared_error(result, test_y)
# print(mse)


#结果输出到文件
train_set = data_preprocess.train
target = data_preprocess.target
model.fit(train_set, target)
test_abbr = data_preprocess.test_abbr
test_abbr.dropna(axis=1,inplace=True)
test_abbr.drop(['survey_time'],axis=1,inplace=True)
result = pd.DataFrame(model.predict(test_abbr),
                      index= test_abbr['id'],columns=["happiness"])
result.to_csv('2.csv')




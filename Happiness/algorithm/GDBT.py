from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import data_preprocess
from sklearn import metrics
from sklearn import model_selection
import pandas as pd
def convert(result):
    for i in range(len(result)):
        if result[i] < 1.5:
            result[i] = 1
        elif result[i] < 2.5:
            result[i] = 2
        elif result[i] < 3.5:
            result[i] = 3
        elif result[i] < 4.5:
            result[i] = 4
        else:
            result[i] = 5

train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
test_x = data_preprocess.test_x
test_y = data_preprocess.test_y
target = data_preprocess.target
# gbc = GradientBoostingClassifier(n_estimators=150,learning_rate=0.21,random_state=10)
gbr = GradientBoostingRegressor(n_estimators=600,loss='ls', learning_rate=0.1,random_state=10)
# gbr30 = GradientBoostingRegressor(n_estimators=120,loss='ls', learning_rate=0.1,random_state=10,max_depth=3)

# gbc.fit(train_x, train_y)
gbr.fit(train_x, train_y)
# gbr30.fit(train_x, train_y)
# result = gbc.predict(test_x)
# mse = metrics.mean_squared_error(result, test_y)
# print(mse)
# result = gbr.predict(test_x)

# cross = model_selection.cross_validate(gbr,train_x,train_y,scoring='neg_mean_squared_error',cv=5)
# cross30 = model_selection.cross_validate(gbr30,train_x,train_y,scoring='neg_mean_squared_error',cv=5)
#
# print('cross: ',cross['test_score'])
# print('30-cross:', cross30['test_score'])
# result = gbr.predict(test_x)
# result30 = gbr30.predict(test_x)
#网格搜索回归提升树
# param1 = {'n_estimators':range(20,200,10)}
# param2 = {'max_depth':range(3, 10)}
# best = model_selection.GridSearchCV(estimator=GradientBoostingRegressor(loss='ls',learning_rate=0.21,
#                                                                  random_state=10),
#                              param_grid=param2, scoring='neg_mean_squared_error',
#                              cv=5)
# best.fit(train_x, train_y)
# print('train best score: ',best.best_score_)
# print('train best para: ', best.best_params_)
# result = best.best_estimator_.predict(test_x)

# mse = metrics.mean_squared_error(result, test_y)
# mse30 = metrics.mean_squared_error(result30, test_y)
# print('mseR :',mse)
# print('mse30R:',mse30)
# convert(result)
# convert(result30)

#gbr n_estimators150 learning_rate0.21  random_state10 scoreR 0.738 scoreC 0.7695
#gbr n_estimators30 learning_rate0.21  random_state10 scoreR 0.764 scoreC  0.8115 GridSearch
# mse = metrics.mean_squared_error(result, test_y)
# mse30 = metrics.mean_squared_error(result30, test_y)
# print('mse :',mse)
# print('mse30:',mse30)

test_abbr = data_preprocess.test_abbr
test_abbr.dropna(axis=1,inplace=True)
test_abbr.drop(['survey_time'],axis=1,inplace=True)
result = gbr.predict(test_abbr)
convert(result)
result = result.astype(int)
result = pd.DataFrame(result,index= test_abbr['id'],columns=["happiness"])
result.to_csv('GDBT.csv')


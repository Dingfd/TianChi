import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
import sklearn
pd.set_option('display.max_columns', None)
train_abbr = pd.read_csv('../data/happiness_train_abbr.csv')
test_abbr = pd.read_csv('../data/happiness_test_abbr.csv')
# train_complete = pd.read_csv('../data/happiness_train_complete.csv', encoding='gbk')
# test_complete = pd.read_csv('../data/happiness_test_complete.csv', encoding='gbk')

#将null值设为农村样本的家庭收入的平均值
train_abbr.loc[train_abbr['family_income'].isnull(),'family_income']\
      = train_abbr.loc[(train_abbr['survey_type'] == 2 )&
                       (train_abbr['work_exper'] == 3)&
                       (train_abbr['family_income']>0)]['family_income'].mean()
train_abbr_set = train_abbr.copy(deep=True)
train_abbr_set.dropna(axis=1,inplace=True)
target = train_abbr_set["happiness"]
train = train_abbr_set.drop(["happiness"], axis=1)
train = train.drop(["survey_time"], axis=1)

train_x, test_x, train_y, test_y = model_selection.train_test_split(train, target,random_state=1)


paramters_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2,4,6,8,10,12,16],
                  'random_state':[0]}
score_MSE = metrics.make_scorer(metrics.mean_squared_error)
best = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=paramters_grid,
                                    cv=100,scoring='neg_mean_squared_error')
best.fit(train_x, train_y)
print(best.best_estimator_)
tree_estimator = best.best_estimator_
print(best.best_score_)
# tree_estimator = tree.DecisionTreeClassifier( criterion='gini',max_depth=2,min_samples_leaf=0.03,random_state=0)
# result = model_selection.cross_validate(tree_estimator, train_x, train_y, cv=20)
# # print(result)


# tree_estimator.fit(train_x, train_y)


back = tree_estimator.predict(test_x)
mse = metrics.mean_squared_error(back, test_y)
print(mse)
print(mse**2*len(test_y))
tree_estimator = tree.DecisionTreeClassifier( max_depth=4,random_state=0)
print(tree_estimator)
tree_estimator.fit(train_x, train_y)
back = tree_estimator.predict(test_x)
mse = metrics.mean_squared_error(back, test_y)
print(mse)
print(mse**2*len(test_y))
#生成上传的结果
# test_abbr.dropna(axis=1,inplace=True)
# test_abbr.drop(['survey_time'],axis=1,inplace=True)
# result = pd.DataFrame(tree_estimator.predict(test_abbr),
#                       index= test_abbr['id'],columns=["happiness"])
# result.to_csv('result.csv')
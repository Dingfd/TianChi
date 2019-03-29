from sklearn.ensemble import AdaBoostClassifier
from sklearn import  ensemble
from sklearn import tree
from sklearn import metrics
import data_preprocess

train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
test_x = data_preprocess.test_x
test_y = data_preprocess.test_y

base_enstimator1 = tree.DecisionTreeClassifier(max_depth=4,random_state=0)
base_enstimator2 = ensemble.RandomForestClassifier(max_depth=18, max_features=22, n_estimators=60, random_state=20)

ada = AdaBoostClassifier(base_enstimator2,n_estimators=25,learning_rate=1)
ada.fit(train_x, train_y)
result = ada.predict(test_x)
mse = metrics.mean_squared_error(result, test_y)
print(mse)
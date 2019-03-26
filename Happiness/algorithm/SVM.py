from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import data_preprocess

train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
test_x = data_preprocess.test_x
test_y = data_preprocess.test_y

# estimator = svm.SVC(kernel='rbf', class_weight='balanced')
# estimator.fit(train_x, train_y)
# result = estimator.predict(test_x)
# print('first : ',metrics.mean_squared_error(result, test_y))

# estimator = svm.SVC(C=2,kernel='rbf', class_weight='balanced')
# estimator.fit(train_x, train_y)
# result = estimator.predict(test_x)
# print('2 : ',metrics.mean_squared_error(result, test_y))

estimator = svm.SVC(C=3,gamma=10,kernel='rbf', class_weight='balanced')
estimator.fit(train_x, train_y)
result = estimator.predict(test_x)
print('3 : ',metrics.mean_squared_error(result, test_y))

# paramater = {
#         'C':[1,3,5,7,9,11,13,15,17,19],
#         'gamma':['auto'],
#         'kernel':['rbf']
#     }
#
# grid = model_selection.GridSearchCV(svm.SVC(), param_grid=paramater,cv=5)
# best = grid.fit(train_x, train_y)
# best_extimator = best.best_estimator_
# back = best_extimator.predict(test_x)
# mse = metrics.mean_squared_error(back, test_y)
# print('bes:',mse)
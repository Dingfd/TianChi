import pandas as pd
from sklearn import model_selection

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
target = train_abbr_set["happiness"]
train = train_abbr_set.drop(["happiness"], axis=1)
# print('train_abbr columns:', train_abbr_set.columns.tolist())
# print('train columns:', train.columns.tolist())
# print('train set :',train.sample(10))
# print('target set :',target.sample(10))
train_x, test_x, train_y, test_y = model_selection.train_test_split(train, target,random_state=1)



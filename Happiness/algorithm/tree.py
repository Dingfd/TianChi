import pandas as pd

pd.set_option('display.max_columns', None)
train_abbr = pd.read_csv('../data/happiness_train_abbr.csv')
test_abbr = pd.read_csv('../data/happiness_test_abbr.csv')
train_complete = pd.read_csv('../data/happiness_train_complete.csv', encoding='gbk')
test_complete = pd.read_csv('../data/happiness_test_complete.csv', encoding='gbk')
# print('删减的训练集：')
# print(train_abbr.info())
# print('删减的测试集')
# print(test_abbr.info())
# print('完整的训练集：')
# print(train_complete.info())
# print('完整的测试集：')
# print(test_complete.info())
# print('train_abbr 5 rows:', train_abbr.sample(5))
# print('test_abbr 5 rows:', test_abbr.sample(5))
print(train_abbr.loc[(train_abbr['survey_type'] == 2 )& (train_abbr['work_exper'] == 3)&(train_abbr['family_income']>0)]
      ['family_income'].mean())
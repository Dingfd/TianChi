import pandas as pd
from sklearn import model_selection

#读取数据
pd.set_option('display.max_columns', None)
train_abbr = pd.read_csv('../data/happiness_train_abbr.csv')
test_abbr = pd.read_csv('../data/happiness_test_abbr.csv')

#将null值设为农村样本的家庭收入的平均值
train_abbr.loc[train_abbr['family_income'].isnull(),'family_income']\
      = train_abbr.loc[(train_abbr['survey_type'] == 2 )&
                       (train_abbr['work_exper'] == 3)&
                       (train_abbr['family_income']>0)]['family_income'].mean()
#去掉null值
train_abbr_set = train_abbr.copy(deep=True)
train_abbr_set.dropna(axis=1,inplace=True)
#选择happiness生成标签
target = train_abbr_set["happiness"]
#去掉happiness和问卷时间生成新的训练数据
train = train_abbr_set.drop(["happiness"], axis=1)
train = train.drop(["survey_time"], axis=1)
#划分训练数据为训练集和测试集
train_x, test_x, train_y, test_y = model_selection.train_test_split(train, target,random_state=1)

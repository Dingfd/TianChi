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
#将调查时间类型转换为datetime类型
train['survey_time'] = pd.to_datetime(train['survey_time'],format='%Y/%m/%d')
#将调查时间设置为调查的年
train['survey_time'] = train['survey_time'].dt.year
#增加年龄特征，调查时间减去出生年
train['age'] = train['survey_time'] - train['birth']
#将年龄分层，高中以下16，高中16-18，大学19-23，入社会24-28,结婚生子29-35，中年危机35-40,壮年40-50,退休前50-60，退休>60
# train.loc[train['age']<16, 'age']= 1
# train.loc[(train['age']>=16) & (train['age']<19),'age'] = 2
# train.loc[(train['age']>=19) & (train['age']<23),'age'] = 3
# train.loc[(train['age']>=23) & (train['age']<29),'age'] = 4
# train.loc[(train['age']>=29) & (train['age']<35),'age'] = 5
# train.loc[(train['age']>=35) & (train['age']<40),'age'] = 6
# train.loc[(train['age']>=40) & (train['age']<50),'age'] = 7
# train.loc[(train['age']>=50) & (train['age']<60),'age'] = 8
# train.loc[train['age']>=60,'age'] = 9

train = train.drop(["survey_time"], axis=1)
#划分训练数据为训练集和测试集
train_x, test_x, train_y, test_y = model_selection.train_test_split(train, target,random_state=1)

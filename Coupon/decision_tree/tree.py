import pandas as pd
from matplotlib import pyplot as plt
import matplotlib_venn as venn
pd.set_option('display.max_columns', None)
train_offline = pd.read_csv('../data/ccf_offline_stage1_train.csv')
train_online = pd.read_csv('../data/ccf_online_stage1_train.csv')
test_offline = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv')

train_off_user = train_offline['User_id'].values
train_on_user = train_online['User_id'].values
test_off_user = test_offline['User_id'].values

set_train_off_user = set(train_off_user)
set_train_on_user = set(train_on_user)
set_test_off_user = set(test_off_user)

TF_to_sf = set_train_off_user - set_train_on_user - set_test_off_user
tf_TO_sf = set_train_on_user - set_train_off_user - set_test_off_user
tf_to_SF = set_test_off_user - set_train_off_user - set_train_on_user
TF_TO_sf = set_train_off_user & set_train_on_user - set_test_off_user
TF_to_SF = set_train_off_user - set_train_on_user & set_test_off_user
tf_TO_SF = set_train_on_user - set_train_off_user & set_test_off_user
TF_TO_SF = set_train_off_user & set_train_on_user & set_test_off_user

venn.venn3(subsets=(len(TF_to_sf), len(tf_TO_sf), len(TF_TO_sf),
                    len(tf_to_SF), len(TF_to_SF), len(tf_TO_SF), len(TF_TO_SF)),
           set_labels=('train_offline_users_num','train_online_users_num', 'test_offline_users_num'))
plt.show()




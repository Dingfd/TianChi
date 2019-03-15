import pandas as pd

pd.set_option('display.max_columns', None)
train_offline = pd.read_csv('../data/ccf_offline_stage1_train.csv')
train_online = pd.read_csv('../data/ccf_online_stage1_train.csv')

print('ddd')
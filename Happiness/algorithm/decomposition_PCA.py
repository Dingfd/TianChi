from sklearn.decomposition import PCA
import data_preprocess

train_set = data_preprocess.train
target = data_preprocess.target
train_x = data_preprocess.train_x
train_y = data_preprocess.train_y
print(len(train_set.columns))
# pca = PCA(n_components=8)
# re = pca.fit_transform(train_set)
# print(re)
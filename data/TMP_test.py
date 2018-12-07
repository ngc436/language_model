import pandas as pd
import deeppavlov

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

elmo = ELMoEmbedder()

# verification = pd.read_csv('/mnt/shdstorage/tmp/classif_tmp/comments_big.csv')
# print(len(verification)) # 15270
previous_weights = "models_dir/model_1542892329.h5"
timing = previous_weights.split('/')[-1].split('_')[-1].split('.')[0]
print(timing)

verification_name = '/mnt/shdstorage/tmp/classif_tmp/comments_big.csv'
path_to_verification = verification_name
data = pd.read_csv(path_to_verification)

print(data.head())
raise ValueError

idx = [2, 4, 6, 8, 16, 32, 64, 128, 256, 432, 678, 975]

y_train_labels = '/mnt/shdstorage/for_classification/y_train_5.csv'
train = pd.read_csv(y_train_labels, header=None).values.tolist()
for i in idx:
    print('true label of %s is %s' % (i, train[i][0]))

path_to_goal_sample = '/mnt/shdstorage/tmp/classif_tmp/comments_big.csv'

data = pd.read_csv(path_to_goal_sample)

idx = [i for i in range(10000, 14000)]

train = data.processed_text.tolist()
texts = data.text.tolist()
print(train[0])

for i in idx:
    print()
    print('====================')
    print()
    print('[%s] text: ' % i)
    print(texts[i])
    print()

# === part 1 ===
# other 139, 119, 101, 295, 293
# long 67, 42, 37, 23, 237
# true 143, 103
# close 13, 140, 137

# === part 2 ===
# other 263, 3481б 11696
# long 2128
# true 3309, 10625
# close 267, 2002, 2025, 2099, 2140

# print()
# data_1 = data.processed_text[[2]].values.tolist()[0]
# print(data.text[[1]].values.tolist()[0])
# print(data_1)
# print()
# data_2 = data.processed_text[[4]].values.tolist()[0]
# print(data.text[[4]].values.tolist()[0])
# print(data_2)
# print()
# data_3 = data.processed_text[[6]].values.tolist()[0]
# print(data.text[[6]].values.tolist()[0])
# print(data_3)
# print()

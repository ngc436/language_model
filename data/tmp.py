import pymystem3

m = pymystem3.mystem.Mystem(mystem_bin='/home/gmaster/mystem')
import re

tags_list = {'<url>', '<date>', '<address>', '<org>', '<name>', '<money>'}
# punct_symbols = {'.', '"', ',', '(', ')', '!', '?', ';', ':', '*','[',']','#','|','\\','%'}

r_vk_ids = re.compile(r'(id{1}[0-9]*)')
r_punct = re.compile(r'[."\[\]/,()!?;:*#|\\%^$&{}~_`=-]')
r_white_space = re.compile(r'\s{2,}')
r_words = re.compile(r'\W+')


def process_punkt(text):
    text = r_punct.sub(" ", text)
    text = r_vk_ids.sub(" ", text)
    text = r_white_space.sub(" ", text)
    return text.strip()


def lemmatize_text(text):
    text = text.lower()
    text = process_punkt(text)
    tokens = r_words.split(text)
    tokens = (x for x in tokens if len(x) >= 2 and not x.isdigit())
    tokens = (m.lemmatize(x)[0] if x not in tags_list else x for x in tokens)
    tokens = (x for x in tokens if x.isalpha() or x in tags_list)
    text = ' '.join(tokens)
    return text


# print(lemmatize_text('Девчата, подскажите, какие у Вас размеры пластин date мм и модели машинок? Хочу на пробу взять date мм, вот и думаю какие размеры нужны будут Вам с данной толщиной'))

# pront validation datas


import pandas as pd
import pickle
import numpy as np

ff = np.zeros((2, 1024))

X_train = pickle.load(open('/mnt/shdstorage/for_classification/elmo/train_v7.pkl', 'rb'))
X_test = pickle.load(open('/mnt/shdstorage/for_classification/elmo/test_v7.pkl', 'rb'))

#print(X_train[0][0])
ff[0] = X_test[0][0]
ff[1] = X_test[1][0]

print(np.append(X_test[0][0], X_test[1][0]))
print(ff)


# ver = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')
# print(ver['edited_text'])
#
# train = pd.read_csv('/mnt/shdstorage/for_classification/train_v7.csv')
# test = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')
#
# print(train['edited_text'])
# print(test['edited_text'])
#
# edited_train = train['edited_text'].tolist()
# edited_test = test['edited_text'].tolist()
#
# # def wrap
# for i in range(300):
#     print('[%s]' % i, edited_test[i])
# # print(edited_train[:100])
# print(edited_test[:100])
# r = [[sent] for sent in edited_train]
# print(r[:3])
#
# print(test['text', 'edited_text', 'processed_text'])

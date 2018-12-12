from fastai.text import *
import pandas as pd
import numpy as np
import html
import re
from collections import *
import pickle

train = pd.read_csv('/mnt/shdstorage/for_classification/train_v7.csv')
test = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('<', ' ').replace('>', ' ').replace('#36;', '$').replace(
        '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('img', ' ').replace('class', ' ').replace(
        'src', ' ').replace('alt', ' ').replace('email', ' ').replace('icq', ' ').replace(
        'href', ' ').replace('mem', ' ').replace('link', ' ').replace('mention', ' ').replace(
        'onclick', ' ').replace('icq', ' ').replace('onmouseover', ' ').replace('post', ' ').replace(
        'local', ' ').replace('key', ' ').replace('target', ' ').replace('amp', ' ').replace(
        'section', ' ').replace('search', ' ').replace('css', ' ').replace('style', ' ').replace(
        'cc', ' ').replace('date', ' ').replace('org', ' ').replace('phone', ' ').replace(
        'address', ' ').replace('name', ' ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df):
    labels = df['label'].values.astype(np.int64)
    texts = df['edited_text'].astype(str)
    texts = list(texts.apply(fixup).values)
    tok = Tokenizer().process_all(texts)
    return tok, list(labels)


tok_trn, trn_labels = get_texts(train)
tok_test, test_labels = get_texts(test)

np.save('/mnt/shdstorage/for_classification/tok_trn_v7_no_ent.npy', tok_trn)
np.save('/mnt/shdstorage/for_classification/tok_test_v7_no_ent.npy', tok_test)

tok_ver = np.load('/mnt/shdstorage/for_classification/tok_test_80_no_ent.npy')

freq = Counter(p for o in tok_trn for p in o)
print(freq.most_common(25))

max_vocab = 80000
min_freq = 2

itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
print(len(itos))

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
test_lm = np.array([[stoi[o] for o in p] for p in tok_test])
ver_lm = np.array([[stoi[o] for o in p] for p in tok_ver])

np.save('/mnt/shdstorage/for_classification/trn_ids_v7_no_ent.npy', trn_lm)
np.save('/mnt/shdstorage/for_classification/test_ids_v7_no_ent.npy', test_lm)
np.save('/mnt/shdstorage/for_classification/ver_ids_v7_no_ent.npy', ver_lm)
pickle.dump(itos, open('/mnt/shdstorage/for_classification/itos_80_no_ent.pkl', 'wb'))

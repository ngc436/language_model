from fastai.text import *
import pandas as pd
import html
import numpy as np
import re
import collections
from collections import *
import pickle

import torch
import torchvision

# train = pd.read_csv('/mnt/shdstorage/for_classification/train_raw.csv')
# val = pd.read_csv('/mnt/shdstorage/for_classification/test_raw.csv')
test = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')

val = pd.read_csv('/mnt/shdstorage/for_classification/test_6_edited_text.csv')
train = pd.read_csv('/mnt/shdstorage/for_classification/train_6_edited_text.csv')

re1 = re.compile(r'  +')


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df):
    labels = df['label'].values.astype(np.int64)
    texts = df['edited_text'].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().process_all(texts)
    return tok, list(labels)


# process sets with tokenizer

tok_trn, trn_labels = get_texts(train)
tok_val, val_labels = get_texts(val)
tok_test, test_labels = get_texts(test)

# save

np.save('/mnt/shdstorage/for_classification/tok_trn_120.npy', tok_trn)
np.save('/mnt/shdstorage/for_classification/tok_val_120.npy', tok_val)
np.save('/mnt/shdstorage/for_classification/tok_test_120.npy', tok_test)

freq = Counter(p for o in tok_trn for p in o)
print(freq.most_common(25))

# save test and train + vocabulary

max_vocab = 120000
min_freq = 2

itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
print(len(itos))

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
test_lm = np.array([[stoi[o] for o in p] for p in tok_test])

np.save('/mnt/shdstorage/for_classification/trn_ids_120.npy', trn_lm)
np.save('/mnt/shdstorage/for_classification/val_ids_120.npy', val_lm)
np.save('/mnt/shdstorage/for_classification/test_ids_120.npy', test_lm)
pickle.dump(itos, open('/mnt/shdstorage/for_classification/itos_120.pkl', 'wb'))

vs = len(itos)
print(vs, len(trn_lm))

itos = pickle.load(open('/mnt/shdstorage/for_classification/itos_120.pkl', 'rb'))
print(itos[:100])

em_sz, nh, nl = 400, 1150, 3

# PRE_PATH = ''
# PRE_LM_PATH = ''
#
# wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
# # to_np ?
# enc_wgts = to_np(wgts['0.encoder.weight'])
# row_m = enc_wgts.mean(0)

# pre lm path

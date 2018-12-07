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

# # train = pd.read_csv('/mnt/shdstorage/for_classification/train_raw.csv')
# # val = pd.read_csv('/mnt/shdstorage/for_classification/test_raw.csv')
# test = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')
#
# val = pd.read_csv('/mnt/shdstorage/for_classification/test_6_edited_text.csv')
# train = pd.read_csv('/mnt/shdstorage/for_classification/train_6_edited_text.csv')
#
# # set(a) & set(b)
#
# re1 = re.compile(r'  +')
#
#
# def fixup(x):
#     x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
#         'nbsp;', ' ').replace('<', ' ').replace('>',' ').replace('#36;', '$').replace(
#         '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
#         '\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
#         ' @-@ ', '-').replace('\\', ' \\ ').replace('img', ' ').replace('class', ' ').replace(
#         'src', ' ').replace('alt', ' ').replace('email', ' ').replace('icq', ' ').replace(
#         'href', ' ').replace('mem', ' ').replace('link', ' ').replace('mention', ' ').replace(
#         'onclick', ' ').replace('icq', ' ').replace('onmouseover', ' ').replace('post', ' ').replace(
#         'local', ' ').replace('key', ' ').replace('target', ' ').replace('amp', ' ').replace(
#         'section', ' ').replace('search', ' ').replace('css', ' ').replace('style', ' ').replace(
#         'cc', ' ').replace('date', ' ').replace('org', ' ').replace('phone', ' ').replace(
#         'address', ' ').replace('name', ' ')
#     return re1.sub(' ', html.unescape(x))
#
#
# def get_text_len(text):
#     return len(text.split())
#
#
# def prepare_set(df, fname='name'):
#     df = df[~df['edited_text'].isna()]
#     df = df[~df['label'].isna()]
#     df['edited_text'] = df['edited_text'].apply(fixup)
#     df['text_len'] = df['edited_text'].apply(get_text_len)
#     df = df[df['text_len'] >= 10]
#     df.to_csv('/mnt/shdstorage/for_classification/%s.csv' % fname, index=None)
#
#
# def get_texts(df):
#     df = df[~df['edited_text'].isna()]
#     df = df[~df['label'].isna()]
#     df['edited_text'] = df['edited_text'].apply(fixup)
#     df['text_len'] = df['edited_text'].apply(get_text_len)
#     df = df[df['text_len'] >= 10]
#     labels = df['label'].values.astype(np.int64)
#     texts = df['edited_text'].astype(str)
#     texts = list(texts.apply(fixup).values)
#     tok = Tokenizer().process_all(texts)
#     return tok, list(labels)
#
#
# prepare_set(test, fname='test_80')
# prepare_set(train, fname='train_80')
# prepare_set(val, fname='val_80')
#
# # process sets with tokenizer
# print('Tokenizing started...')
# tok_trn, trn_labels = get_texts(train)
# tok_val, val_labels = get_texts(val)
# tok_test, test_labels = get_texts(test)
#
# # save
# print('Saving to files...')
# np.save('/mnt/shdstorage/for_classification/tok_trn_80_no_ent.npy', tok_trn)
# np.save('/mnt/shdstorage/for_classification/tok_val_80_no_ent.npy', tok_val)
# np.save('/mnt/shdstorage/for_classification/tok_test_80_no_ent.npy', tok_test)
#
# freq = Counter(p for o in tok_trn for p in o)
# print(freq.most_common(25))
#
# # save test and train + vocabulary
#
# max_vocab = 80000
# min_freq = 2
#
# itos = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
# itos.insert(0, '_pad_')
# itos.insert(0, '_unk_')
#
# stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
# print(len(itos))
#
# trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
# val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
# test_lm = np.array([[stoi[o] for o in p] for p in tok_test])
#
# np.save('/mnt/shdstorage/for_classification/trn_ids_80_no_ent.npy', trn_lm)
# np.save('/mnt/shdstorage/for_classification/val_ids_80_no_ent.npy', val_lm)
# np.save('/mnt/shdstorage/for_classification/test_ids_80_no_ent.npy', test_lm)
# pickle.dump(itos, open('/mnt/shdstorage/for_classification/itos_80_no_ent.pkl', 'wb'))
#
# vs = len(itos)
# print(vs, len(trn_lm))

itos = pickle.load(open('/mnt/shdstorage/for_classification/itos_80_no_ent.pkl', 'rb'))
print(itos[:100])

vs = len(itos)

em_sz, nh, nl = 400, 1150, 3

# lm4.h5  lm_enc4.h5

PRE_LM_PATH = '/mnt/shdstorage/for_classification/weights/lm4.h5'
lm_itos = '/mnt/shdstorage/for_classification/itos.pkl'

wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = np.array(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)
itos2 = pickle.load(open(lm_itos, 'rb'))
stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})

new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i, w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r >= 0 else row_m

wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

# pre lm path

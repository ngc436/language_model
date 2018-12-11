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
        'address', ' ').replace('name', ' ').replace('\n', '').replace('\r', '').replace(
        '|', '').replace('id', ' ').replace('[', '').replace(']', '').replace('span')
    return re1.sub(' ', html.unescape(x))


def fixup_2(x):
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
        'cc', ' ').replace('\n', '').replace('\r', '').replace('|', '').replace('id', ' ').replace(
        '[', '').replace(']', '')

    splitted = x.split()
    print(splitted[:3])
    if 'phone' in splitted[:3]:
        part_1 = ' '.join(splitted[:3])
        part_1 = part_1.replace('phone', '')
        part_2 = ' '.join(splitted[3:])
        x = '%s %s' % (part_1, part_2)

    x = re1.sub(' ', html.unescape(x))
    return re1.sub(' ', html.unescape(x))


def get_text_len(text):
    return len(text.split())


class TextTokenizer:

    def __init__(self, voc_size, min_freq):
        self.voc_size = voc_size
        self.min_freq = min_freq
        self.itos = None
        self.stoi = None

    def get_text_len(text):
        return len(text.split())

    def save_set(self, df, fname='name'):
        df = self._clean(df)
        path = '/mnt/shdstorage/for_classification/%s.csv' % fname
        df.to_csv(path, index=None)
        print('Dataframe is saved to %s' % path)

    def _clean(self, df):
        df = df[~df['edited_text'].isna()]
        df = df[~df['label'].isna()]
        df['edited_text'] = df['edited_text'].apply(fixup)
        df['text_len'] = df['edited_text'].apply(get_text_len)
        df = df[df['text_len'] >= 10]
        return df

    def get_texts(self, df):
        df = self._clean(df)
        labels = df['label'].values.astype(np.int64)
        texts = df['edited_text'].astype(str)
        return texts, labels

    def get_tokens(self, df):
        texts, labels = self.get_texts(df)
        tok = Tokenizer().process_all(texts)
        return tok, list(labels)

    def ids(self, tok_trn):
        print(tok_trn[:10])
        freq = Counter(p for o in tok_trn for p in o)
        print(freq.most_common(25))
        self.itos = [o for o, c in freq.most_common(self.voc_size) if c > self.min_freq]
        self.itos.insert(0, '_pad_')
        self.itos.insert(0, '_unk_')
        self.stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(self.itos)})


ver = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')

print(ver['edited_text_old'])

ver['edited_text'] = ver['edited_text_old'].apply(fixup_2)
ver.to_csv('/mnt/shdstorage/for_classification/new_test.csv')

# train = pd.read_csv('/mnt/shdstorage/for_classification/train_v7.csv')
# test = pd.read_csv('/mnt/shdstorage/for_classification/test_v7.csv')
# ver = pd.read_csv('/mnt/shdstorage/for_classification/new_test.csv')
#
# tt = TextTokenizer(voc_size=80000, min_freq=2)
#
# tok_trn, trn_labels = tt.get_tokens(train)
# tok_test, test_labels = tt.get_tokens(test)
# tok_ver, ver_labels = tt.get_tokens(ver)
#
# np.save('/mnt/shdstorage/for_classification/tok_trn_80_no_ent_v7.npy', tok_trn)
# np.save('/mnt/shdstorage/for_classification/tok_ver_80_no_ent_v7.npy', tok_ver)
# np.save('/mnt/shdstorage/for_classification/tok_test_80_no_ent_v7.npy', tok_test)
#
# tt.ids(tok_trn)
#
# trn_lm = np.array([[tt.stoi[o] for o in p] for p in tok_trn])
# ver_lm = np.array([[tt.stoi[o] for o in p] for p in tok_ver])
# test_lm = np.array([[tt.stoi[o] for o in p] for p in tok_test])
#
# np.save('/mnt/shdstorage/for_classification/trn_ids_80_no_ent_v7.npy', trn_lm)
# np.save('/mnt/shdstorage/for_classification/ver_ids_80_no_ent_v7.npy', ver_lm)
# np.save('/mnt/shdstorage/for_classification/test_ids_80_no_ent_v7.npy', test_lm)
# pickle.dump(tt.itos, open('/mnt/shdstorage/for_classification/itos_80_no_ent_v7.pkl', 'wb'))
#
# raise ValueError
#
# itos = pickle.load(open('/mnt/shdstorage/for_classification/itos_80_no_ent.pkl', 'rb'))
# print(itos[:100])
#
# vs = len(itos)
#
# em_sz, nh, nl = 400, 1150, 3
#
# # lm4.h5  lm_enc4.h5
#
# PRE_LM_PATH = '/mnt/shdstorage/for_classification/weights/lm4.h5'
# lm_itos = '/mnt/shdstorage/for_classification/itos.pkl'
#
# wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
# enc_wgts = np.array(wgts['0.encoder.weight'])
# row_m = enc_wgts.mean(0)
# itos2 = pickle.load(open(lm_itos, 'rb'))
# stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
#
# new_w = np.zeros((vs, em_sz), dtype=np.float32)
# for i, w in enumerate(itos):
#     r = stoi2[w]
#     new_w[i] = enc_wgts[r] if r >= 0 else row_m
#
# wgts['0.encoder.weight'] = T(new_w)
# wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
# wgts['1.decoder.weight'] = T(np.copy(new_w))
#
# # pre lm path

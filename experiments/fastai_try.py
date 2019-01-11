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

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
from collections import OrderedDict

set_session(session)


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
        '|', '').replace('id', ' ').replace('[', '').replace(']', '').replace('span', ' ')

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


class Tokenizator:

    def __init__(self, voc_size=60000, min_freq=2):
        self.tt = TextTokenizer(voc_size=voc_size, min_freq=min_freq)

    # set mode to 'train' to create ids set
    def prepare_set(self, set_fname, mode=None):
        set = pd.read_csv(set_fname, encoding='utf-8')
        tok, labels = self.tt.get_tokens(set)
        if mode == 'train':
            self.tt.ids(tok)
        lm = np.array([[self.tt.stoi[o] for o in p] for p in tok])
        return tok, labels, lm


# the function to convert h5 model to pth
def convert(path_to_old_model, path_to_save_converted_model):
    """
    path_to_old_model is the path to old model
    and
    path_to_save_converted_model is the path where the converted model is stored
    """
    old_wgts = torch.load(path_to_old_model, map_location=lambda storage, loc: storage)
    new_wgts = OrderedDict()
    new_wgts['encoder.weight'] = old_wgts['0.encoder.weight']
    new_wgts['0.encoder.weight'] = old_wgts['0.encoder.weight']
    new_wgts['encoder_dp.emb.weight'] = old_wgts['0.encoder_with_dropout.embed.weight']
    new_wgts['rnns.0.weight_hh_l0_raw'] = old_wgts['0.rnns.0.module.weight_hh_l0_raw']
    new_wgts['rnns.0.module.weight_ih_l0'] = old_wgts['0.rnns.0.module.weight_ih_l0']
    new_wgts['rnns.0.module.weight_hh_l0'] = old_wgts['0.rnns.0.module.weight_hh_l0_raw']
    new_wgts['rnns.0.module.bias_ih_l0'] = old_wgts['0.rnns.0.module.bias_ih_l0']
    new_wgts['rnns.0.module.bias_hh_l0'] = old_wgts['0.rnns.0.module.bias_hh_l0']
    new_wgts['rnns.1.weight_hh_l0_raw'] = old_wgts['0.rnns.1.module.weight_hh_l0_raw']
    new_wgts['rnns.1.module.weight_ih_l0'] = old_wgts['0.rnns.1.module.weight_ih_l0']
    new_wgts['rnns.1.module.weight_hh_l0'] = old_wgts['0.rnns.1.module.weight_hh_l0_raw']
    new_wgts['rnns.1.module.bias_ih_l0'] = old_wgts['0.rnns.1.module.bias_ih_l0']
    new_wgts['rnns.1.module.bias_hh_l0'] = old_wgts['0.rnns.1.module.bias_hh_l0']
    new_wgts['rnns.2.weight_hh_l0_raw'] = old_wgts['0.rnns.2.module.weight_hh_l0_raw']
    new_wgts['rnns.2.module.weight_ih_l0'] = old_wgts['0.rnns.2.module.weight_ih_l0']
    new_wgts['rnns.2.module.weight_hh_l0'] = old_wgts['0.rnns.2.module.weight_hh_l0_raw']
    new_wgts['rnns.2.module.bias_ih_l0'] = old_wgts['0.rnns.2.module.bias_ih_l0']
    new_wgts['rnns.2.module.bias_hh_l0'] = old_wgts['0.rnns.2.module.bias_hh_l0']
    new_wgts['1.decoder.bias'] = old_wgts['1.decoder.weight']

    torch.save(new_wgts, path_to_save_converted_model + 'lm4.pth')

convert('/home/gmaster/projects/negRevClassif/pretrained_lm/lm4.h5', '/home/gmaster/projects/negRevClassif/pretrained_lm/')


# print(ver['edited_text_old'])
#
# ver['edited_text'] = ver['edited_text_old'].apply(fixup_2)
# ver.to_csv('/mnt/shdstorage/for_classification/new_test.csv')

# tokenizer = Tokenizator()
#
train_fname = '/home/gmaster/projects/negRevClassif/source_datasets/train_v7.csv'
test_fname = '/home/gmaster/projects/negRevClassif/source_datasets/test_v7.csv'
ver_fname = '/home/gmaster/projects/negRevClassif/source_datasets/new_test.csv'
#
# #
# tok_trn, trn_labels, trn_lm = tokenizer.prepare_set(train_fname, mode='train')
# tok_test, test_labels, test_lm = tokenizer.prepare_set(test_fname)
# tok_ver, ver_labels, ver_lm = tokenizer.prepare_set(ver_fname)
#
# np.save('/home/gmaster/projects/negRevClassif/source_datasets/trn_ids_60_tt.npy', trn_lm)
# np.save('/home/gmaster/projects/negRevClassif/source_datasets/test_ids_60_tt.npy', test_lm)
# np.save('/home/gmaster/projects/negRevClassif/source_datasets/ver_ids_60_tt.npy', ver_lm)
# pickle.dump(tokenizer.tt.itos, open('/home/gmaster/projects/negRevClassif/source_datasets/itos_60_tt.pkl', 'wb'))
#

# itos = tokenizer.tt.itos
itos = pickle.load(open('/home/gmaster/projects/negRevClassif/source_datasets/itos_60_tt.pkl', 'rb'))

print(itos[:100])
#

vs = len(itos)
em_sz, nh, nl = 400, 1150, 3

#
# # lm4.h5  lm_enc4.h5

PRE_LM_PATH = '/home/gmaster/projects/negRevClassif/pretrained_lm/lm4'
lm_itos = '/home/gmaster/projects/negRevClassif/pretrained_lm/itos'

# wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
# enc_wgts = np.array(wgts['0.encoder.weight'])
# row_m = enc_wgts.mean(0)
# itos2 = pickle.load(open(lm_itos, 'rb'))
# stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})
#
# new_w = np.zeros((vs, em_sz), dtype=np.float32)
# for i, w in enumerate(itos.pkl):
#     r = stoi2[w]
#     new_w[i] = enc_wgts[r] if r >= 0 else row_m
#
# wgts['0.encoder.weight'] = torch.from_numpy(new_w)
# wgts['0.encoder_with_dropout.embed.weight'] = torch.from_numpy(np.copy(new_w))
# wgts['1.decoder.weight'] = torch.from_numpy(np.copy(new_w))

# opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

train = pd.read_csv(train_fname)
train_df = train[['label', 'edited_text']]
train_df = train_df.rename(columns={'edited_text': 'text'})

test = pd.read_csv(test_fname)
test_df = test[['label', 'edited_text']]
test_df = test_df.rename(columns={'edited_text': 'text'})
print('Datasets are ready!')

data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=test_df, path="")
data_clas = TextClasDataBunch.from_df(path='', train_df=train_df, valid_df=test_df,
                                      vocab=data_lm.train_ds.vocab, bs=32)
print('Bunches are ready!')
#
# data_lm.save('/home/gmaster/projects/negRevClassif/data/')
# data_clas.save('/home/gmaster/projects/negRevClassif/data/')

# data_lm = TextLMDataBunch.load('/home/gmaster/projects/negRevClassif/data/')
# data_clas = TextClasDataBunch.load('/home/gmaster/projects/negRevClassif/data/', bs=32)

# language model
learn = language_model_learner(data_lm, pretrained_fnames=[PRE_LM_PATH, lm_itos], drop_mult=0.7)
print(learn.fit_one_cycle(1, 1e-2))

# classifier
learn = text_classifier_learner(data_clas, drop_mult=0.7)
learn.fit_one_cycle(1, 1e-2)

preds, targets = learn.get_preds()
predictions = np.argmax(preds, axis=1)
pd.crosstab(predictions, targets)


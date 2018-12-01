# from fastai.learner import *
#
# import torchtext
# from torchtext import vocab, data
# from torchtext.datasets import language_modeling
#
# from fastai.rnn_reg import *
# from fastai.rnn_train import *
# from fastai.nlp import *
# from fastai.lm_rnn import *

import dill as pickle
import spacy


from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality

from urllib import request as urlrequest

# proxy_host = 'proxy.ifmo.ru:3128'    # host and port of your proxy
#
# req = urlrequest.Request(url)
# req.set_proxy(proxy_host, 'http')


path = '/mnt/shdstorage/for_classification'
df = pd.read_csv(path/'new_test.csv', header=None)
df.head()

data_lm = TextLMDataBunch.from_csv(path, 'new_test.csv')
data_clas = TextClasDataBunch.from_csv(path, 'new_test.csv', vocab=data_lm.train_ds.vocab, bs=42)




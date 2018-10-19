import gensim.downloader as api
from gensim.models import  KeyedVectors
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import nltk
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def convert_tag(tag):
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return 'NOUN'
    if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return 'VERB'
    if tag in ['RB', 'RBR', 'RBS']:
        return 'ADV'
    if tag in ['JJ', 'JJR', 'JJS']:
        return 'ADJ'
    return 'trash'

# COMP,
# 'ADJ', 'ADV', 'VERB'
def tag_mappings(tag):
    if tag in ['NOUN']:
        return 'NOUN'
    if tag in ['ADJF','ADJS']:
        return 'ADJ'
    if tag in ['VERB','INFN','PRTF','PRTS','GRND']:
        return 'VERB'
    if tag in ['ADVB']:
        return 'ADV'
    if tag in ['CCONJ']:
        return
    return 'X'

path_to_model = '/tmp/web_0_300_20.bin'

word_list = ['кот', 'идти', 'здание']
res = [(x, morph.parse(x)[0].tag.POS) for x in word_list]

#res = nltk.pos_tag(['кот', 'идти', 'здание'])
#print(res)

tokens_res = []
for i in res:
    tokens_res.append(i[0] +'_'+ tag_mappings(i[1]))
print(tokens_res)

# len 300 of each vector in w2v
# if word not in the model - random vector (?)
model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
print(len(model.wv['кот_NOUN']))
for word, distance in model.most_similar(tokens_res[2]):
    print(u"{}:{:.3f}".format(word, distance))


from gensim.models import FastText

print('Starting to load fasttext embeddings...')
path_to_fasttext_emb = '/tmp/wiki.ru.bin'
print('Done!')

ft_model = FastText.load_fasttext_format(path_to_fasttext_emb)

print(ft_model.wv['снег'])

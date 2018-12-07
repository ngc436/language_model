import pickle
import numpy as np

# verif, test, train
path_to_batch = '/mnt/shdstorage/for_classification/elmo/processed_elmo_batch_%s.pkl'

tok_trn = np.load('/mnt/shdstorage/for_classification/tok_trn_80_no_ent.npy')
tok_val = np.load('/mnt/shdstorage/for_classification/tok_val_80_no_ent.npy')
tok_test = np.load('/mnt/shdstorage/for_classification/tok_test_80_no_ent.npy')


def get_initial_position():
    raise NotImplementedError


for i in range(2):
    batch = pickle.load(open('/mnt/shdstorage/for_classification/elmo/processed_elmo_batch_%s.pkl' % i, 'rb'))
    print(batch)


def yield_batches():
    # TODO: to do this
    yield batch

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session

set_session(session)
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

MODELS_DIR = 'models_dir'


def save_model_to_json(model, fname):
    model_json = model.to_json()
    with open('%s/%s.json' % (MODELS_DIR, fname), "w") as json_file:
        json_file.write(model_json)
    print("Model is saved as json")

class Base_model:

    # def __init__(self):


    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

class LSTM(Base_model):

    def __init__(self):
        self.max_sentence_len = 50
        self._initialization()

    def _initialization(self):

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,
                            weights=pretrained_weights))
        model.add(LSTM(units=embedding_size))
        model.add(Dense(uniits=vocab_size))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        print(model.summary())


    # def generate_word(self):
    #
    # def sam
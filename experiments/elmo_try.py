# https://tfhub.dev/google/elmo/2
# https://towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440

import tensorflow as tf
import tensorflow_hub as hub


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
["первое предложение", "второе предложение"],
signature="default",
as_dict=True)["elmo"]

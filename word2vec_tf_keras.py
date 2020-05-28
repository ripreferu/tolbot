# -*- encoding: utf-8 -*-
from datetime import datetime
import tensorflow as tf
import pickle
import class_texte
import numpy as np
# unpickle liste_de_docment_traites
# unpickle vocabulaire mot --> indices
liste_docments_traités = pickle.load(open("docment_traites.p", 'rb'))
test_2 = pickle.load(open("dico_mot_index.p", 'rb'))
"""Plongement de mots """
for doc in liste_docments_traités:
    for item in doc:
        # print(item)
        object_texte = item[0]
        # print(object_texte)
        object_texte.__one_hot__()
        # print(object_texte.oh)
liste_oh = [txt.oh for doc in liste_docments_traités for (
    txt, nature, ID) in doc]

# en clair le one hot c'est utilsier le dico test_2 pour remplacer les listes de string par une liste de nombres

# On pourrait stocker ces informations dans le type de texte mais le vocabulaire pose problème
# En effet le vocabulaire est une variable de la classe de Texte
# Cette variable est issue de l'analyse de l'ensemble des texte Cf la fonction vocabulary()


# après il va falloir faire des regroupements
# aka générer des couples de nombre entrées sorties
# possibilité de faire du slicing
#import tensorflow as tf
Dim_voc = len(set(test_2))
X, Y = [], []
window_size = 5
dim_plongement = 2
long_oh = [len(i) for i in liste_oh]
vocab_size = len(test_2)
max_longueur = max(long_oh)
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(
    liste_oh, maxlen=max_longueur, padding="post")
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
    vocab_size)
plt_mot = tf.keras.layers.Embedding(vocab_size, dim_plongement, name='plt_mot')
# inputs
w_inputs = tf.keras.Input(shape=(1,), dtype='int32')
w = plt_mot(w_inputs)
# context

c_inputs = tf.keras.Input(shape=(1,), dtype='int32')
c = plt_mot(c_inputs)
o = tf.keras.layers.Dot(axes=2)([w, c])
o = tf.keras.layers.Reshape((1,), input_shape=(1, 1))(o)
o = tf.keras.layers.Activation("sigmoid")(o)
SkipGram = tf.keras.Model(inputs=[w_inputs, c_inputs], outputs=o)
SkipGram.summary()
SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
tensorboard_callback = tf.compat.v1.keras.callbacks.TensorBoard(
    log_dir=logdir, write_graph=True)

X = []
Y = []
for _ in range(5):
    for item in liste_oh:
        couples, labels = tf.keras.preprocessing.sequence.skipgrams(
            item, vocab_size, window_size, sampling_table=sampling_table)
        for couple in couples:
            X.append(couple)
        for label in labels:
            Y.append(label)
X_new = [np.array(x) for x in zip(*X)]
SkipGram.fit(X_new, Y, batch_size=10, epochs=5, verbose=1,
             callbacks=[tensorboard_callback]
             )
matrice_plgt = SkipGram.get_layer("plt_mot").get_weights()[0]

"""à exporter via pickle ou numpy """
np.save(open("modele_plgt.npy", 'wb'), matrice_plgt)

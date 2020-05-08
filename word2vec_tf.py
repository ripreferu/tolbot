

###########################################################################################
# ATTENTION CE MODÈLE EST CHRONOPHAGE, IL EST CONSERVÉ PAR BUT PÉDAGOGIQUE UNIQUEMENT     #
###########################################################################################

"""

Ce modèle réalisé sous tensorflow version 1 n'est pas du tout optimisé
il prend des heures (littéralement!) possibilité de jouer sur la batch size
le problèmes vient probablement d'un manipulation trop conséquentes des entrées
vecteurs de dimension vocab*1 ==> alors que ce sont des vecteurs creux 1 seul coordonnées non nulles
conséquences on se trainent pleins de coéfficients inutiles dans la matrice

Solution 
utiliser l'implémentation keras fournit
exploiter l'implémentation dans gensim
"""
from nltk.corpus import stopwords
import stanza
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import numpy as np
import tensorflow as tf
import pandas as pd
import class_xml
from scipy.sparse import csr_matrix
import random

""" import le module class_texte """
""" Recuperer le liste_docment_traites"""
""" Récupérer le vocabulaire"""

"""Plongement de mots """
for doc in liste_docments_traités:
    for item in doc:
        #print(item)
        object_texte=item[0]
        #print(object_texte)
        object_texte.__one_hot__()
        #print(object_texte.oh)
liste_oh=[txt.oh for doc in liste_docments_traités for (txt,nature,ID) in doc]

# en clair le one hot c'est utilsier le dico test_2 pour remplacer les listes de string par une liste de nombres

#On pourrait stocker ces informations dans le type de texte mais le vocabulaire pose problème
# En effet le vocabulaire est une variable de la classe de Texte
# Cette variable est issue de l'analyse de l'ensemble des texte Cf la fonction vocabulary()


## après il va falloir faire des regroupements
## aka générer des couples de nombre entrées sorties
## possibilité de faire du slicing 

"""générer des clouples de données """
Taille_fenêtre=5
liste_de_couple=[]
for each_doc in liste_oh:
    for indice,wordx in enumerate(each_doc):
        for voisin in each_doc[max(indice-Taille_fenêtre,0): min(indice+Taille_fenêtre,len(each_doc)+1)]:
            if voisin!=wordx:
                liste_de_couple.append([wordx,voisin])

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
Dim_voc=len(set(test_2))
X,Y=[],[]
# X entrée
# Y sortie
def from_numbers_to_array(indice):
    ans=np.zeros(Dim_voc)
    ans[indice]=1
    return ans
    pass
for item  in  liste_de_couple:
    X.append(from_numbers_to_array(item[0]))
    Y.append(from_numbers_to_array(item[1]))
X_train=np.asarray(X)
Y_train=np.asarray(Y)
# window_size=5
# dim_plongement=2
# long_oh=[len(i) for i in liste_oh]
# vocab_size=len(test_2)
# max_longueur=max(long_oh)
# padded_docs=tf.keras.preprocessing.sequence.pad_sequences(liste_oh, maxlen=max_longueur,padding="post")
# sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
# plt_mot=tf.keras.layers.Embedding(vocab_size,dim_plongement,name='plt_mot')
# # inputs
# w_inputs = tf.keras.Input(shape=(1,),dtype='int32')
# w=plt_mot(w_inputs)
# # context

# c_inputs= tf.keras.Input(shape=(1,),dtype='int32')
# c=plt_mot(c_inputs)
# o=tf.keras.layers.Dot(axes=2)([w,c])
# o=tf.keras.layers.Reshape((1,),input_shape=(1,1))(o)
# o=tf.keras.layers.Activation("sigmoid")(o)
# SkipGram=tf.keras.Model(inputs=[w_inputs,c_inputs],outputs=o)
# SkipGram.summary()
# SkipGram.compile(loss='binary_crossentropy',optimizer='adam')
# from datetime import datetime
# now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir="tf_logs"
# logdir="{}/run-{}/".format(root_logdir,now)
# tensorboard_callback=tf.compat.v1.keras.callbacks.TensorBoard(log_dir=logdir,write_graph=True)
# #writer=tf.summary.FileWriter(logdir,graph=)
# X=[]
# Y=[]
# for _ in range(5):
#     for item in liste_oh:
#         couples,labels=tf.keras.preprocessing.sequence.skipgrams(item, vocab_size, window_size,sampling_table=sampling_table)
#         for couple in couples:
#             X.append(couple)
#         for label in labels:
#             Y.append(label)
# X_new=[np.array(x) for x in zip(*X)]
# SkipGram.fit(X_new,Y,batch_size=10,epochs=5,verbose=1,
#              callbacks=[tensorboard_callback]
# )
# matrice_plgt=SkipGram.get_layer("plt_mot").get_weights()[0]
# tfidf=TfidfTransformer()
# matrice_tfidf=tfidf.fit_transform(Matrice_term_doc)
# matrice_BM25F=BM_25F(Matrice_term_doc)
# liste_modèle=[matrice_tfidf,matrice_BM25F,matrice_plgt]
# with open("modele_mat.npy",'wb') as f:
#     for mat in liste_modèle:
#         np.save(f,mat)
# import pickle

# pickle.dump(test_2,open("dico_mot_index.p", 'wb'))
# pickle.dump(test_dico, open("dico_index_mot.p",'wb'))
# pickle.dump(liste_id,open("liste_id_doc.p", 'wb'))
# pickle.dump(liste_docments_traités,open("docment_traites.p",'wb'))
# pickle.dump(liste_doc_xml,open("liste_doc_xml.p","wb"))
# pickle.dump(stop_words,open("stop_words.p","wb"))
# pickle.dump(p,open("reg_exp.p","wb"))
x=tf.compat.v1.placeholder(tf.float32,shape=(None,Dim_voc),name="entree")

y_label=tf.compat.v1.placeholder(tf.float32,shape=(None,Dim_voc), name="sortie")

""" -* IMPORTANT CHOIX DE LA DIMENSION DU PLONGEMENT *- """
""" finalement c'est la même que du clustering il va falloir la reduction de dimensionalité """

DIM_plongement=2
""" La litterature recommande un nombre dimension entre 100 et 1000"""
# FIRST HIDDEN LAYER
with tf.compat.v1.variable_scope("premiere_couche"):
    W1=tf.Variable(tf.random.normal([Dim_voc,DIM_plongement]),name="poids")
    b1=tf.Variable(tf.random.normal([1]),name="biais") # Bias
    Couche_cachee=tf.add(tf.matmul(x,W1),b1,name="neurone_lineaire")

""" SORTIE """

with tf.compat.v1.variable_scope("Softmax"):
    W2 = tf.Variable(tf.random.normal([DIM_plongement,Dim_voc]),name="poids")
    b2 = tf.Variable(tf.random.normal([1]),name="biais")
    logit=tf.add(tf.matmul(Couche_cachee,W2),b2, name="neurone_lineaire")
    pred=tf.compat.v1.nn.softmax(logit,name="softmax")

fonction_objectif=tf.math.reduce_mean(
#    tf.compat.v1.losses.softmax_cross_entropy(labels=y_label,logits=logit))
    -tf.reduce_sum(y_label*tf.math.log(pred),axis=[1]))
#

optimizer=tf.compat.v1.train.MomentumOptimizer(0.05,0.9, use_nesterov=True)
train_op=optimizer.minimize(fonction_objectif)
from datetime import datetime
now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir,now)

summary_loss=tf.compat.v1.summary.scalar("Xentropy",fonction_objectif)
summary_writer=tf.compat.v1.summary.FileWriter(logdir,tf.compat.v1.get_default_graph())

# TRAINING -- ENTRAÎNEMENT
sess=tf.compat.v1.Session()
saver=tf.compat.v1.train.Saver()
init=tf.compat.v1.global_variables_initializer()
sess.run(init)
iteration = 1000 # 50 000
for i in range(iteration):    
    sess.run(train_op, feed_dict={x: X_train, y_label:Y_train})
    if i% 100 == 0: 
        print(str(i/iteration*100)+"%: iteration n°"+str(i)+" fonction objectif", sess.run(fonction_objectif, feed_dict={x:X_train,y_label:Y_train}))
        summary_str=summary_loss.eval(session=sess,feed_dict={x:X_train,y_label:Y_train})
        summary_writer.add_summary(summary_str,i)
    if i% 1000 ==0:
        save_path=saver.save(sess, "./modelV3.ckpt")
save_path=saver.save(sess, "./final_modelV3.ckpt")
summary_writer.close()
repz_2D=sess.run(W1+b1)
#print(type(repz_2D))
# import pickle
# pickle.dump(repz_2D,open('vecteur_mots.p',"wb"))


import matplotlib.pyplot as plt
from gensim.sklearn_api import Text2BowTransformer
from gensim.models import LdaModel, LsiModel
import pickle
import numpy as np
from scipy.sparse import load_npz
# from coclust.visualization import plot_max_modularities
from coclust.evaluation.internal import best_modularity_partition
import wordcloud as wc

term_doc = load_npz("matrice_BM25F.npz")
cluster_range = range(2, 10)
coclust_mod, modularities = best_modularity_partition(
    term_doc, cluster_range, n_rand_init=1)
# plot_max_modularities(modularities, cluster_range)
nbre_cluster = modularities.index(max(modularities))+cluster_range[0]

liste_doc_traite = pickle.load(open("docment_traites.p", 'rb'))
corpus = [objet_texte.txt_lemma_str for doc in liste_doc_traite for (
    objet_texte, nature, docID) in doc]
bow = Text2BowTransformer()
corpus_2 = bow.fit_transform(corpus)
model_lda = LdaModel(corpus_2, num_topics=nbre_cluster,
                     id2word=bow.gensim_model)
topics_lda = model_lda.get_topics()
model_lsi = LsiModel(corpus_2, num_topics=nbre_cluster,
                     id2word=bow.gensim_model)
dico_id2mot = bow.gensim_model.id2token


def subsampling(*arg):
    m = max(arg)
    rep = [False]*len(arg)
    ind = arg.index(m)
    rep[ind] = m
    return rep


def mot_clustering(model, dico):
    i = 0
    topics = model.get_topics()
    reponse = []
    sujet = []
    matrice = np.array([np.array(subsampling(*k)) for k in topics.transpose()])
    for topic in topics:
        for mot_id in dico.keys():
            # print("mot id0", mot_id, "i", i)
            if bool(matrice[mot_id, i]):
                sujet.append((dico[mot_id], matrice[mot_id, i]))
        sujet.sort(key=lambda t: t[1])
        reponse.append(sujet[:])
        sujet = []
        i += 1
    return reponse


liste_lsi = mot_clustering(model_lsi, dico_id2mot)
liste_lda = mot_clustering(model_lda, dico_id2mot)


# def top(liste, K=10):
#     '''donne les K premier mots du thèmes '''
#     rep = ""
#     for topics in liste:
#         rep += "nouveau sujet: \n"
#         rep += " ".join([i[0] for i in topics[:K]])
#         rep += " \n "
#     return rep


# # print(top(liste_lsi))
# # print(top(liste_lda))


def nuage(model, dico):
    #  Extraction des fréquences
    topics = model.get_topics()
    liste = []
    matrice = np.array([np.array(subsampling(*k)) for k in topics.transpose()])
#    print(matrice.shape)
    for i in range(len(topics)):
        di = {}
        for mot_id in dico.keys():
            if bool(matrice[mot_id, i]):
                di[dico[mot_id]] = matrice[mot_id, i]
        # print(di)
        nuage_obj = wc.WordCloud(background_color='white')
        liste.append(nuage_obj.generate_from_frequencies(di))
        del nuage_obj
    return liste


nuages_lsi = nuage(model_lsi, dico_id2mot)
nuages_lda = nuage(model_lda, dico_id2mot)


def Affichage_nuage(liste_nuages):
    for cloud in liste_nuages:
        plt.clf()
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

# Affichage_nuage(nuages_lsi)

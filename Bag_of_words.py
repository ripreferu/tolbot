import pickle
from scipy.sparse import csr_matrix
from scipy import sparse
# import class_texte
import logging
import numpy as np


def sac_de_mots(voc, liste_des_str, stopwords):
    """ renvoie la matrice terme de document
    avec voc le dico vocabulaire qui fait le lien index-> mots
    besoin des stopwords pour éviter les problèmes"""
    from sklearn.feature_extraction.text import CountVectorizer
    Vectorizer = CountVectorizer(vocabulary=voc, stop_words=stopwords)
    matrice_term_doc = Vectorizer.transform(liste_des_str)
    return matrice_term_doc


def tf_idf(matrice_term_doc):
    """applique le tf_idf à la matrice termes documents données en entrée """
    from sklearn.feature_extraction.text import TfidfTransformer
    Transformation = TfidfTransformer()
    matrice_tfidf = Transformation.fit_transform(matrice_term_doc)
    return matrice_tfidf


def BM_25F(matrice, liste_doc_traités):
    """ prends la matrice terme documents et calcule le BM25F
    creer un matrice/sparse matrice de même dimension
    pour chaque ligne / documents de la matrice TD
    on peut calculer la longueur de document
    Sommes des indice du vecteur docments ("ligne" de la matrice)

    META: il faut Toutes les longueur de ts les documents.
    Car on a besoin de la longueur moyenne

    Opération sur les colonnes / les mots du corpus
    Compter le nombre de documents où le mot apparaît -- c'est pas si simple
    #Partie obscure
    #Compter le nombre de documents pertinent contenant le terme i
    #Compter le nombre de documents pertinents dans la correction
    """
    # A BESOIN DE LISTE_DOCMENTS_TRAITÉS
    assert isinstance(matrice, csr_matrix)
    nb_doc, long_voca = matrice.shape
    W = csr_matrix((nb_doc, long_voca), dtype=np.float_)
    liste_doc_plate = [b for item in liste_doc_traités for b in item]
    print(nb_doc)
    print(len(liste_doc_plate))
    poids = []
    for i in range(nb_doc):
        nature = liste_doc_plate[i][1]
        if nature == "Autre" or nature == "Doc":
            poids.append(1)
        else:  # if nature=="Section":
            poids.append(3)
    poids = np.array(poids)
    matrice_pond = csr_matrix((nb_doc, long_voca), dtype=np.float_)
    for i in range(nb_doc):
        matrice_pond[i, :] = matrice[i, :]*poids[i]
    longueur_doc = np.array([matrice_pond[i, :].sum() for i in range(nb_doc)])
    mean_len_doc = np.mean(longueur_doc)
    nb_doc_terme = [matrice[:, j].count_nonzero() for j in range(long_voca)]
    b, k = 0.75, 1.2
    # print(mean_len_doc)
    """ tentative d'optimisation en se limitant au valeurs non nulles
 de la matrices CSR """
    (liste_i, liste_j) = matrice.nonzero()
    liste_i = list(liste_i)
    liste_j = list(liste_j)
    logging.info("Comparaison {} pour {}".format(
        nb_doc*long_voca, len(liste_j)))
    for compteur in range(len(liste_i)):
        i, j = liste_i[compteur], liste_j[compteur]
        if nb_doc_terme[j] != 0:
            para_a = (matrice_pond[i, j]*(k+1))/(
                k*((1-b)+b * (longueur_doc[i]/mean_len_doc))
                + matrice_pond[i, j])
            para_b = np.log((nb_doc-nb_doc_terme[j]+0.5)/(nb_doc_terme[j]))
            W[i, j] = para_a*para_b
    return W


docment_traites = pickle.load(open("docment_traites.p", 'rb'))
input_liste = [object_texte.txt_lemma_str for doc in docment_traites for (
    object_texte, nature, ID) in doc]
stopwords = pickle.load(open("stop_words.p", 'rb'))
voc = pickle.load(open("dico_mot_index.p", 'rb'),)
matrice_term_doc = sac_de_mots(voc, input_liste, stopwords)
matrice_tfidf = tf_idf(matrice_term_doc)
matrice_BM25F = BM_25F(matrice_term_doc, docment_traites)
sparse.save_npz("matrice_tfidf.npz", matrice_tfidf)
sparse.save_npz("matrice_BM25F.npz", matrice_BM25F)

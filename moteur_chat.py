import pickle
import numpy as np
import logging
import random
from scipy import sparse
import class_xml
import class_texte
from pyemd import emd
# logging.basicConfig(filename="chatbot.log",level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('chatlog.log')
formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
dico_mot_index = pickle.load(open("dico_mot_index.p", "rb"))
dico_index_mot = pickle.load(open("dico_index_mot.p", "rb"))
list_id_doc = pickle.load(open("liste_id_doc.p", "rb"))
list_doc_xml = pickle.load(open("liste_doc_xml.p", "rb"))
doc_traites = pickle.load(open("docment_traites.p", "rb"))

tfidf = sparse.load_npz("matrice_tfidf.npz")
BM25F = sparse.load_npz("matrice_BM25F.npz")
plongement = np.load(open("modele_plgt.npy", 'rb'), allow_pickle=True)
dico_doc = dict(enumerate(
    [DocID for doc in doc_traites for (objet_texte, nature, DocID) in doc]))
retour_f = []


def protocole_sac_de_mot(matrice, Req, dico_doc, dico_mots):
    """ la matrice définit le modèle sac de mots basée sur une matrice terme
    documents:
    la réquête est lématisé sous une forme d'une liste de lemmes
    le dico_doc fait le lien numéro de ligne ->documents
    le dico_mots fait le lien lemme -> numéro de la colonne de la matrice"""
    assert len(Req) > 0, 'la requête est vide'
    nbre_mots = len(dico_mots)
    vec_Req = np.zeros((nbre_mots, 1))
    # Création du vecteur_requête au sens sac de mots
    for lemme in Req:
        indice = None
        try:
            indice = dico_mots[lemme]
        except KeyError:
            logger.warning(" le lemme {} n'est pas dans le vocabulaire du chatbot.\
            Ce mot ne sera pas pris compte dans la requête.".format(lemme))
        if indice is not None:
            vec_Req[indice, 0] += 1
            del indice
        # les formules de tf idf ou BM25 sont pensées au niveau d'un corpus
        # ont elle un sens pour un seul document?
        # en soit cela devrait suffir
    # Fin de la création du vecteur_requête au sens sac de mots.
    nbre_doc = len(dico_doc)
    score = []
    long_req = len(Req)
    for ligne in range(nbre_doc):
        vec_doc = matrice[ligne, :]
        logger.debug("dimension de vecteur document:{}".format(vec_doc.shape))
        logger.debug("dimenson de vecteur réquête:{}".format(vec_Req.shape))
        calcul = (vec_doc@vec_Req)/long_req
        calcul = calcul.reshape(1).tolist()
        score.append((dico_doc[ligne], calcul[0]))
        # problème sur la ligne il me faut le docID associé à la ligne111
    # la liste de score donne le score de la réquête vis_a_vis du corpus1
    # on pourrait retourner le score
    # But I don't
    score.sort(key=lambda item: item[1], reverse=True)
    return score


def wmd(doc1, doc2, matrice_plgt):
    """ Soit doc_vec2 et doc_vec1 deux doc encodées à la one_hot sur un\
 vocabulaire connu
    code fortement inspiré et réécrit depuis la fonction wmdistance de gensim
    Une Référence leurs reviennent de droits
    """
    if len(doc2)*len(doc1) == 0:
        logger.debug(
            "l'un des 2 documents est vide --> distance infinie entre\
 les 2 docs")
        return float('inf')
    # isoler les mots utilisées dans les 2 docs et les compter
    doc_fusion = doc1+doc2
    dico = {}
    for i in set(doc_fusion):
        dico[i] = doc1.count(i)+doc2.count(i)
    # la longueur du vocabulaire s'applique ici uniquement aux 2 documents
    long_voca = len(dico)
    docset1 = set(doc1)
    docset2 = set(doc2)
    """calcul de la matrices des distances
    ces distances servent pour après """
    distance_matrix = np.zeros((long_voca, long_voca), dtype=np.double)
    for i, t1 in enumerate(dico.keys()):
        if t1 not in docset1:
            continue
        for j, t2 in enumerate(dico.keys()):
            if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                continue
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                np.sum((matrice_plgt[t1]-matrice_plgt[t2])**2))
    if np.sum(distance_matrix) == 0.0:
        logger.info("la matrice des distance ne contient que des zeros")
        print("k")
        return float('inf')

    def nbow(doc):
        d = np.zeros(long_voca, dtype=np.double)
        list_freq = [doc.count(i)
                     for i in set(doc_fusion)]  # fréquence des mots
        doc_len = len(doc)
        for idx, freq in enumerate(list_freq):
            d[idx] = freq/float(doc_len)
        return d
    d1 = nbow(doc1)
    d2 = nbow(doc2)
    return emd(d1, d2, distance_matrix)


def protocole_plongement_mot(matrice, req, dico_mot_index, liste_oh):
    """ req la réquête sous forme d'une liste de lemmmes
    la liste des oh sous la forme (doc_oh,docID)
    """
    assert len(req) > 0, "la requete est vide"
    """Etape 1 transformation de la réquête
 pour la poser un encodage one_hot """
    req_l = []
    for lemme in req:
        try:
            req_l.append(dico_mot_index[lemme])
        except KeyError:
            logging.warning(" le lemme {} n'est pas dans le vocabulaire du chatbot.\
            Ce mot ne sera pas pris compte dans la requête.".format(lemme))
    score = []
    for doc, docID in liste_oh:
        score.append((docID, wmd(req_l, doc, matrice)))
    score.sort(key=lambda item: item[1])
    return score


def test_models(user_entry):
    """ Pour une réquête données, la fonctiion renvoie les score des \
différents modèles Nota Bene les scores sont  des liste des couples \
(DocID, valeur numérique du score)
    """
    lemme_list = class_texte.lemma(user_entry)  # necessaire à tout les modèles
    logging.debug("{},{}".format(user_entry, lemme_list))
    class_texte.Texte.__voc__ = dico_mot_index
    for doc in doc_traites:
        for object_texte, natrure, doc_id in doc:
            object_texte.__one_hot__()
    liste_oh = [(i.oh, doc_id)
                for doc in doc_traites for (i, nature, doc_id) in doc]
    score_plgt_mot = protocole_plongement_mot(
        plongement, lemme_list, dico_mot_index, liste_oh)
#    print(len(score_plgt_mot))
    score_BM25F = protocole_sac_de_mot(
        BM25F, lemme_list, dico_doc, dico_mot_index)
    score_tfidf = protocole_sac_de_mot(
        tfidf, lemme_list, dico_doc, dico_mot_index)
    return score_plgt_mot, score_BM25F, score_tfidf


def docID2text(DocID):
    """interprête le DocID en object 'class xml'
    doc ID est un couple (id du noeud du doc, id du doc)
    -----
    renvoie élément (class_xml), class_xml.document
    """
    # Etape 1 retrouve le document
    node_id, Doc_id = DocID
    for Doc in list_doc_xml:
        if Doc.id == Doc_id:
            Document = Doc
            logger.debug("on a retrouvé le document!\n \
            il se trouve à {}".format(Document.chemin.as_posix))
    # print(Document)
    # Etape 2 retourver la partie du document concerné
    tree = Document.treelib
    tree_node = tree.get_node(node_id)
    elt = tree_node.data
    logger.debug("l'élément trouvé a les\
 caractéristiques suivantes: \
    est une branche: {} \t est une feuille: {} \t Nature exacte: {} \
\t niveau{} \t    \
    ".format(tree.is_branch(node_id), tree_node.is_leaf(),
             tree_node.tag, tree.level(node_id)))
    return elt, Document


##################################
# Code volé au Chatbot de Sophie #
##################################
""" Initialisation """

liste_salutations = ["salut", "hey", "bonjour", "ave cesar", "yo", "sal'ss"]
salutation_rep = ["Salut !", "Hey !", "Bonjour !"]

liste_remerciements = ['merci', 'merci beaucoup', 'cimer']
remerciement_rep = ['De rien !', 'C\'était un plaisir !',
                    'Pas de soucis !', 'Aucun problème !',
                    'Ravi d\'avoir pu t\'aider !']


def salut(phrase):
    ''' Fonction permettant de répondre à une salutation de l'utilisateur.
    La fonction prend en paramètre une chaine de caractère, de type string.
    Après vérification de si cette chaine contient des éléments
présents dans la liste_salutations, elle retourne (ou non) une réponse
 au hasard dans la liste de reponses.
    '''
    for mot in phrase.split():
        if mot.lower() in liste_salutations:
            return random.choice(salutation_rep)


def remerciement(phrase):
    ''' Fonction permettant de répondre à un remerciement de l'utilisateur.
    La fonction prend en paramètre une chaine de caractère, de type string.
    Après vérification de si cette chaine contient des éléments présents \
dans la \
    liste_remerciements, elle retourne (ou non) une réponse au hasard \
dans la liste de reponses.
    '''
    for mot in phrase.split():
        if mot.lower() in liste_remerciements:
            return random.choice(remerciement_rep)
######################
# FIN DU VOL DE CODE #
######################


def moteur():
    """ moteur du chatbot"""
    signal = True
    motd = "Bonjour, je m'appelle TolBot ! \n \
    Je suis disponible pour répondre à tes questions sur le tolérancement. \n \
    N'hésite pas à me poser tes questions ! \n \
    Pour partir il te suffit de dire 'salut'!"
    print(motd)
    # Regarder l'architecture du moteur
    while signal:  # Attente de l'entrée utilsateur
        requete_user = input('Utilisateur : ')
        requete_user = requete_user.lower()
#        liste_req_user=requete_user.split()
        if 'salut' not in requete_user:
            rem = remerciement(requete_user)
            sal = salut(requete_user)
            if bool(rem):  # Faux si vide
                # signal=False
                print('Tolbot: {}'.format(rem))
            elif bool(sal):  # Faux si vide
                print('Tolbot : {}'.format(sal))
            else:
                scores = list(test_models(requete_user))
                model_liste = ["plongement de mots",
                               "sac de mots BM25F", "sac de mot tfidf"]
                for num, each_model in enumerate(scores):
                    logger.info("------------------------")
                    logger.info(
                        "Résultat selon le modèle {}: ".format(
                            model_liste[num]))
                    # print(each_model[0])
                    logger.info("------------------------")
                    for i in range(2):
                        elt, doc = docID2text(each_model[i][0])
                        logger.info("{0} element {1} \n {2}".format(
                            i, doc.Titre, type(elt)))
                        if isinstance(elt, class_xml.section):
                            logger.info('{}'.format(elt.Titre))
                        if isinstance(elt, class_xml.paragraph):
                            logger.info(elt.texte)
                classement_B, classement_A1, classement_A2 = scores[0][:],\
                    scores[1][:], scores[2][:]
                class_B = {}
                class_A1 = {}
                class_A2 = {}
                for doc, score in classement_B:
                    element_list = (doc, score)
                    class_B[doc] = classement_B.index(element_list)+1
                for doc, score in classement_A1:
                    el = (doc, score)
                    class_A1[doc] = classement_A1.index(el)+1
                for doc, score in classement_A2:
                    el = (doc, score)
                    class_A2[doc] = classement_A2.index(el)+1
                print(len(class_A1) == len(class_A2) == len(class_B))
                class_C = []
                if len(class_A1) == len(class_A2) == len(class_B):
                    for doc in class_A1.keys():
                        class_C.append((doc,
                                        (class_A1[doc] + class_B[doc])/2))
                class_C.sort(key=lambda item: item[1])
                elt, doc = docID2text(class_C[0][0])
                retour_f.append(doc)
                if isinstance(elt, class_xml.paragraph) or isinstance(
                        elt, class_xml.note):
                    print(elt.texte)
                if isinstance(elt, class_xml.image):
                    print('Attention la réponse est une image')
                if isinstance(elt, class_xml.document):
                    print('La réponse est un document')
                if isinstance(elt, class_xml.section):
                    print("la réponse est une section")
                    print(elt.Titre)
        else:
            signal = False
            return("Tolbot: Au revoir")


moteur()
print(retour_f)

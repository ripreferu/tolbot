# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 16:00:13 2019

@author: pierr_000
"""


''' COMMENTAIRES :
    Ce fichier comporte un certain nombre de programme et de fonctionnalités :
        >> chatbot() permet une conversation classique sans les améliorations
        avec le chatbot
        >> entrainement() permet d'entrainerle chatbot
        >> TolBot() permet de converser avec un chatbot améliorer

    Merci de commencer par exécuter les fonctionnalités en fin de fichier (l. 717)
'''
# %%
''' Importation des bibliothèques nécessaires '''
# import dictionnaire_synV7 as dico #importation du dictionnaire des synonymes
# IDF = Inverse Document Frequency : permet de représenter les mots en vecteurs
# Cosine Similarity : mesure de similarité entre 2 vecteurs non nuls
""" Paramétrage de la bibliothèque stanfordnlp"""
# etape à suivre
# stanfordnlp.download('fr')

from itertools import product
from numpy import where
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import class_xml as extracted_data
from nltk.corpus import stopwords
import stanfordnlp
import random  # permet de choisir une réponse au hasard dans une liste de possibilité
nlp = stanfordnlp.Pipeline(processors="tokenize,pos,lemma", lang='fr')
# %%
""" Tokenisation """


def tokenisation_phrase(texte):
    liste_resultats = []
    phrase_a_tokeniser = nlp(texte)
    for phrase in phrase_a_tokeniser.sentences:
        resultat = ""
        for mot in phrase.words:
            resultat += mot.text
            resultat += " "
        liste_resultats.append(resultat)
    return liste_resultats


def tokenisation_mot(texte):
    liste_resultats = []
    phrase_a_tokeniser = nlp(texte)
    for phrase in phrase_a_tokeniser.sentences:
        # resultat=""
        for mot in phrase.words:
            # resultat+=mot.text
            #resultat+=" "
            liste_resultats.append(mot.text)
    return liste_resultats


#
""" Suppression des StopWords """
stop_words = set(stopwords.words('french'))
""" Traitement  """


def traitement(texte):
    """ l'intégralité du traitement est maintenant réalisé par
    stanfordnlp. # plus de bricolage infernal"""
    texte = texte.lower()
    doc = nlp(texte)
    answer = []
    for phrases in doc.sentences:
        for word in phrases.words:
            if word.lemma not in stop_words:
                # print(word.lemma)
                answer.append(word.lemma)
    return answer


# %%
""" Lecture de données d'un corpus """


def traitement_section(texte_sections, texte_traite_hier=[]):
    """Fonction_recursive avec la liste texte_sections 
    texte_traite_hier étant une liste

    """
    for sous_sections in texte_sections:
        if isinstance(sous_sections, str):
            # print(sous_sections)
            texte_traite_hier += [traitement(sous_sections)]
        elif isinstance(sous_sections, list):
            texte_traite_hier += [traitement_section(
                sous_sections, texte_traite_hier)]
    return texte_traite_hier


#texte = open('fichier_sans_extentions.txt','r',encoding="utf8")
texte = extracted_data.test.Texte
# corpus_traite=traitement_section(texte)
# corpus=texte.read()
# corpus_traite=traitement(corpus)
# token_phr = tokenisation_phrase(corpus)
# texte.close()

# %%
""" Analyse lexico-syntaxique """


def sans_accents(mot):
    ''' Fonction permettant d'enlever les accents d'un mot.
    Elle prend en entrée un mot sous la forme d'une chaine de caractères
    de type string.
    Elle retourne la même chaine sans accents.
    '''
    nouv_mot = ''
    for i in mot:
        if ord(i) in [232, 233, 234, 235]:
            nouv_mot += 'e'
        elif ord(i) in [224, 226]:
            nouv_mot += 'a'
        elif ord(i) in [238, 239]:
            nouv_mot += 'i'
        elif ord(i) == 244:
            nouv_mot += 'o'
        elif ord(i) == 251:
            nouv_mot += 'u'
        else:
            nouv_mot += i
    return nouv_mot

# %%


def calcul_similarite(mot1, mot2):
    ''' Fonction permettant de calculer le coefficient de similarité entre 2 mots.
    Cependant il n'est pas aisé de calculer une similarité entre 2 mots seulement,
    nous allons donc rajouter une partie de phrase neutre avant chaque mot (la même).
    Elle prend en paramètre les 2 mots dont on souhaite calculer le coefficient de 
    similarité.
        /!\ ATTENTION : mot1 = mot d'origine    mot2 = synonyme dont on veut calculer 
        la similarité avec le mot d'origine
    Elle retourne la valeur de ce coefficient.
    '''
    test = ('Que veut dire {}'.format(mot1), 'Que veut dire {}'.format(mot2))
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(test)
    result_cos = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    return result_cos[0][-1]
# %%


def ajout_syn(mot, dic_syns):
    ''' Fonction permettant d'ajouter un synonyme d'un mot ainsi que sa distance 
    (par calcul du coefficient de similarité) à ce mot.
    Elle prend en argument le mot dont on souhaite ajouter les synonymes au 
    dictionnaire sous forme de chaine de caractère de type string.
    Elle retourne le dictionnaire des synonymes complété.
    '''
    mot.lower()
    if not mot in dic_syns:
        syns = dico.syn_mot(mot)  # liste des synonymes du mot

        # calcul du coefficient de similarité entre chaque mot et son synonyme
        liste_simil = []
        for k in syns:
            liste_simil.append((k, calcul_similarite(mot, k)))

        dic_syns[mot] = liste_simil
    return dic_syns
# %%


def synonyme_liste(liste_noms):
    ''' Fonction permettat de créer la liste des synonymes.
    Cette fonction sera exécutée à chaque réouverture du fichier,
    une seule fois dans la séance.
    Elle prend en entrée la liste de noms dont on cherche les synonymes.
    Elle retourne le dictionnaire des synonymes complété.
    '''
    dic_syns = {}
    l = len(liste_noms)
    for k in liste_noms:
        print(l, ' ', k)
        ajout_syn(k, dic_syns)
        l = l-1
    return dic_syns
# %%


def synonyme(mot):
    ''' Fonction permettant de récupérer la liste des synonymes d'un mot.
    Elle prend en paramètre une chaine de caractère de la forme string.
    Elle retourne la liste de ses synonymes grace à la librairie dictionnaire.
    '''
    mot = mot.lower()
    mot = sans_accents(mot)
    if mot in dic_syns:
        return ([(mot, 1.)]+dic_syns[mot])
    else:
        return [(mot, 1.)]
# %%


def paraphrase(phrase):
    ''' Fonction permettant de retourner une liste de paraphrases de la phrase rentrée 
    (cad. une liste de phrases synonymes de la phrase placée en paramètre).
    Elle prend en paramètre une phrase de la forme 'il était une fois dans l'ouest.'.
    Elle retourne une liste de paraphrase (soit une liste de strings) de la forme 
    ['paraphrase1','paraphrase2'...].
    La fonction comporte plusieurs étapes :
            etape 1 >> traitement de la phrase pour obtenir une liste de mots de la forme 
            ['mot1','mot2',...]
            etape 2 >> obtention d'une liste contenant les listes des synonymes de la liste 
            précédente [['syn11','syn12',...],['syn21','syn22',...],...]
            etape 3 >> génération de toutes les paraphrases possibles stockées dans une liste 
            ['paraphrase1','paraphrase2','paraphrase1','paraphrase3',...]
            etape 4 >> détermination du nombre X de possibles pour notre phrase 
            (X = nbre élement pour syn1 * nbre éléments syn2 *...) par calcul du produit 
            cartésien entre mes différents ensemble (ici mes différentes liste de syn) et 
            ainsi extraire les X derniers termes de la liste L
            etape 5 >> traitement de la liste obtenues de la forme (((syn1,syn2),syn3),syn4)
            pour obtenir 'syn1 syn2 syn3 syn4'
            etape 6 >> association de chaque paraphrase au produit des distances 
            (des différents synonymes la composant avec  leurs mots d'origine)
    '''
    syns_pond = []  # liste contenant les listes des synonymes des mots de la phrase
    # avec leurs pondérations
    syns = []  # liste contenant seulement les synonymes des mots de la phrase
    ponds = []  # liste contenant seulement les pondérations des mots de la phrase

    '''etape 1'''
    phrase = phrase.lower()
    liste_lemm = traitement(phrase)

    '''etape 2'''
    for k in liste_lemm:
        syns_pond.append(synonyme(k))
    #syns_pond = [[(mot1,1.),(syn11,pond1),...,(syn1n,pond1n)],...,[(motn,1.),...,(synnn,pondnn)]]

    for k in syns_pond:
        syn = []
        pond = []
        for i in k:
            # on récupère les synonymes sans leurs pondérations
            syn.append(i[0])
            pond.append(i[1])
        syns.append(syn)
        ponds.append(pond)

    '''etape 3'''
    L_syn = []  # liste contenant les différentes paraphrases non traitées
    L_pond = []  # liste contenant les pondérations

    # initialisation de la boucle pour remplir la liste
    for x in product(syns[0], syns[1]):
        # L_syn et ainsi l'utliser par la suite
        L_syn.append(x)

    # initialisation de la boucle pour remplir la liste
    for x in product(ponds[0], ponds[1]):
        # L_pond et ainsi l'utliser par la suite
        L_pond.append(x)

    for k in range(2, len(syns)):
        # utilisation de la liste L_syn précédemment formée
        for x in product(L_syn, syns[k]):
            # pour effectuer le produit cartésien
            L_syn.append(x)
        # utilisation de la liste L_pond précédemment
        for y in product(L_pond, ponds[k]):
            # formée pour effectuer le produit cartésien
            L_pond.append(y)

    '''etape 4'''
    X = 1  # nombre de permutations possibles
    for k in syns:
        X *= len(k)
    N_syn = L_syn[len(L_syn)-X:]  # sélection des X dernière phrases
    # (celles d'avant ont servis pour la construction)
    # de même, les 2 listes ainsi crées ont la même longueur
    N_pond = L_pond[len(L_pond)-X:]

    '''etape 5'''
    M_syn = ["" for k in range(
        len(N_syn))]  # création d'une liste M de même longeur que N
    M_pond = []

    # première boucle pour obtenir les paraphrases par traitement de texte
    for k in range(len(N_syn)):
        for i in str(N_syn[k]):
            if i != '(' and i != ')' and i != ',' and i != "'":  # nettoyage des éléments de N
                M_syn[k] += i

    # deuxième boucle pour obtenir les pondérations totales
    for k in N_pond:
        pond = 1.
        while type(k) == tuple and type(k[1]) == float:
            pond = pond * k[1]
            k = k[0]
        M_pond.append(pond*k)

    # retourne une liste de X éléments contenant les différentes possibilités de
    # paraphrases avec leurs pondérations associées
    return [(M_syn[k], M_pond[k]) for k in range(len(M_syn))]


# %%
""" Génération de réponses """


def reponse(requete_utilisateur):
    ''' Fonction permettant la génération de réponse à une requête de l'utilisateur à partir
    du calcul du coefficient de similarité entre 2 chaines de caractères.
    Elle prend en paramètre la requête saisie par l'utilisateur, de la forme chaine de 
    caractère de type string 'mot1 mot2 mot3 ...' ainsi que le maximum de la mesure
    de similarité.
    Elle retourne un string, chaine de caractères, de réponse.
    La fonction se décompose en plusieurs étapes :
        etape 1 >> traitement de la requete (génération de paraphrases)
            on ajoute chaque élément de la liste requetes à un corpus, puis on tokenize 
            en phrase
            cette manipulation va nous permettre de mettre un poids sur chaque valeur 
            de Val_liste
        etape 2 >> calcul du coefficient de similarité pour chaque élément de CORPUS_TOKENS :
            IDF = TfidfVectorizer().fit transform(CORPUS_TOKENS[i]) permet de transformer 
            le corpus en une matrice servant pour le calcul du coefficient de similarité
            cosine_similarity(IDF_liste[i][-1],IDF_liste[i]) calul ce coefficient entre le 
            dernier élement de la matrice (à savoir la requête) et l'ensemble de la matrice 
        etape 3 >> permet de déterminer la liste des coefficients les plus élevés pour 
        chaque élément de CORPUS_TOKENS :
            on choisit Flat[-2] car Flat[-1] correspond au coefficient de similarité entre 
            la requête et la requête (vaut toujours 1)
        etape 4 >> on va pondérer chaque terme de la liste Val_liste
        etape 5 >> permet de déterminer l'indice du coeff le plus élevé à partir duquel on 
        récupère la phrase dans la base de données ayant le même indice (cad la réponse) 
        et on la retourne SSI la valeur du coefficient retenue est différente de 0
    '''

    '''etape 1'''
    requetes = paraphrase(
        requete_utilisateur)  # on considère que l'utilisateur émet une
    # requête ne contenant qu'une phrase
    # le corpus a été traité de la même manière que
    corpus = ' '.join(corpus_traite)
    # la requête : token°, stop_words, étiquetage, lemm°
    # on isole les mots du corpus qui sont dons la requête ?
    CORPUS = [corpus for k in requetes]
    # print(CORPUS)
    for k in range(len(requetes)):
        CORPUS[k] += ' '
        CORPUS[k] += requetes[k][0]
    CORPUS_TOKENS = []
    # meme problème qu'avec la réponse_tables
    for i in CORPUS:
        CORPUS_TOKENS.append(tokenisation_phrase(i))

    '''etape 2'''
    IDF_Vect = TfidfVectorizer()  # transforme un doc en matrice TF-IDF
    IDF_liste = []
    for i in CORPUS_TOKENS:
        IDF = IDF_Vect.fit_transform(i)  # fit_transform(corpus)
        IDF_liste.append(IDF)
    Coeff_sim_liste = [cosine_similarity(
        IDF_liste[i][-1], IDF_liste[i]) for i in range(len(IDF_liste))]

    '''etape 3'''
    Flat_liste = [Coeff_sim_liste[i].flatten()
                  for i in range(len(Coeff_sim_liste))]
    # Coeff_sim = array([[x,y,...]]) et la commande flatten permet d'obtenir array([x,y,z...])
    for k in Flat_liste:
        k.sort()  # trie des différentes listes de la valeur la plus faible à la plus élevée
    Val_liste = [Flat_liste[k][-2] for k in range(len(Flat_liste))]
    # retourne la valeur la plus élevée du coefficient de similarité

    '''etape 4'''
    Val_liste_pond = []
    for k in range(len(Val_liste)):
        Val_liste_pond.append(Val_liste[k]*requetes[k][1])

    '''etape 5'''
    # N permet de déterminer quel élément de requête a la meilleur valeur de similarité
    N = Val_liste_pond.index(max(Val_liste_pond))
    # n : indice de la valeur max de Val_liste
    n = where(Coeff_sim_liste[N] == Val_liste[N])[1][0]
#    print("mesure de similarité = ",Val_liste_pond[N])

    if(Val_liste[N] == 0) or (n > len(token_phr)):
        return ['Je suis désolée mais je n\'ai pas compris le sens de votre demande. Pouvez vous reformuler votre question ?', 0.5]  # HACK
    else:
        # on récupère la phrase et non la phrase traitée pour répondre à la question
        return token_phr[n], Val_liste_pond[N]


# %%
""" Initialisation """

liste_salutations = ["salut", "hey", "bonjour", "ave cesar", "yo", "sal'ss"]
salutation_rep = ["Salut !", "Hey !", "Bonjour !"]

liste_remerciements = ['merci', 'merci beaucoup', 'cimer']
remerciement_rep = ['De rien !', 'C\'était un plaisir !',
                    'Pas de soucis !', 'Aucun problème !', 'Ravi d\'avoir pu t\'aider !']


def salut(phrase):
    ''' Fonction permettant de répondre à une salutation de l'utilisateur.
    La fonction prend en paramètre une chaine de caractère, de type string.
    Après vérification de si cette chaine contient des éléments présents dans la 
    liste_salutations, elle retourne (ou non) une réponse au hasard dans la 
    liste de reponses.
    '''
    for mot in phrase.split():
        if mot.lower() in liste_salutations:
            return random.choice(salutation_rep)


def remerciement(phrase):
    ''' Fonction permettant de répondre à un remerciement de l'utilisateur.
    La fonction prend en paramètre une chaine de caractère, de type string.
    Après vérification de si cette chaine contient des éléments présents dans la 
    liste_remerciements, elle retourne (ou non) une réponse au hasard dans la 
    liste de reponses.
    '''
    for mot in phrase.split():
        if mot.lower() in liste_remerciements:
            return random.choice(remerciement_rep)


# %%
""" Script Chatbot """

aide_prof = {}


def chatbot():
    ''' Fonction permettant de chatter avec le bot.
    Elle ne prend pas de paramètre en entrée et permet de retourner une réponse de type 
    string à une requête que l'utilisateur rentrera dans la console (input()).
    Elle fonctionne à l'aide de balise. On initialise la balise comme étant True et tant 
    qu'elle le reste le programme continue.
    Cette fonction fait appel aux fonctions précédentes. 
    Rapportez vous aux doc-strings de ces fonctions pours plus d'information.
    '''

    signal = True
    print("TolBot : Bonjour, je m'appelle TolBot!")
    print("Je suis disponible pour répondre à tes questions sur le tolérancement.")
    print("N'hésite pas à me poser tes questions !")
    print("Pour partir, il te suffit de dire 'salut' !")

    while(signal == True):
        requete_utilisateur = input("Utilisateur : ")
        requete_utilisateur = requete_utilisateur.lower()
        if 'salut' not in requete_utilisateur:
            if remerciement(requete_utilisateur) != None:
                signal = False
                return("TolBot : {}".format(remerciement(requete_utilisateur)))
            else:
                if salut(requete_utilisateur) != None:
                    print("TolBot : {}".format(salut(requete_utilisateur)))
                else:
                    rep = reponse(requete_utilisateur)[0]
                    print("TolBot : {}".format(rep))
                    # stockage de la requête pour le transmettre à l'enseignant
                    aide_prof[requete_utilisateur] = rep
        else:
            signal = False
            return("TolBot : A la revoyure !")


# %%
""" Amélioration : retour d'une liste questions / réponses vers l'expert """


def retour_prof():
    ''' Fonction permettant de retourner la liste des questions posées au Chatbot.
    Elle ne prend pas de paramètre en entrée.
    Elle retourne un dictionnaire présentant une requête posée au Chatbot associée
    à la réponse fournit par se dernier.
    Seule les requêtes dont les réponses seront cherchées dans la base de données
    seront considérées.
    '''
    return aide_prof


""" Amélioration : mode entrainement """


def entrainement():
    ''' Fonction permettant d'entrainer le chatbot en vérifiant la pertinence des
    réponses retourner à une requête utilisateur et décidant, ou non, d'en implémenter
    une plus pertinente.
    Elle ne prend pas de paramètre en entrée et retourne une réponse de type string à
    chaque requete rentrée.
    '''

    signal = True
    print("TolBot : Bonjour, bienvenue dans le mode entrainement ! ")
    print("Le fonctionnement de ce mode est le suivant : tu poses des questions sur le tolrancement géométrique au Chatbot et évalue la pertinence des réponses.")
    print("Si la réponse n'est pas pertinente, tu devras alors rentrer la bonne réponse.")
    print("Si tu souhaites arrêter le mode entrainement il te suffit de dire 'stop'.")

    while(signal == True):
        requete_utilisateur = input("Pose moi une question : ")
        requete_utilisateur = requete_utilisateur.lower()
        if 'stop' in requete_utilisateur:
            signal = False
            return ("TolBot : Merci pour ton aide !")
        else:
            rep = reponse(requete_utilisateur)[0]
            print("TolBot : {}".format(rep))
            avis = input(" Ma réponse était-elle suffisamment pertinente ? ")
            if avis.lower() == "non":
                nouvelle_rep = input("Rentre une réponse plus pertinente : ")
                table_QR = open("table_QR.txt", 'a')
                table_QR.write(requete_utilisateur + ";" + nouvelle_rep + ";")
                table_QR.close()


""" Ajout du mode entrainement au chatbot """


def repartition(table_QR):
    ''' Fonction permettant de séparer les questions et les réponses en 2 listes.
    Elle prend en entrée le nom du fichier contenant la table sous la forme d'une
    chaine de caractère de type string.
    Elle retourne 2 listes de questions et réponses : QU et REP.
    '''
    fichier = open("{}.txt".format(table_QR), "r", encoding="iso-8859-1")
    table = fichier.read()
    fichier.close()

    QU, REP = [], []
    decomp = table.split(';')
    n = 0
    while n < len(decomp):
        if n == 0:
            QU.append(decomp[n])
        elif n % 2 == 1:
            REP.append(decomp[n])
        else:
            QU.append(decomp[n])
        n += 1
    return QU, REP


def reponse_table(requete_utilisateur):
    '''  Fonction permettant la génération de réponse à une requête de l'utilisateur à partir
    du calcul du coefficient de similarité entre 2 chaines de caractères.
    Elle prend en paramètre la requête saisie par l'utilisateur, de la forme chaine de 
    caractère de type string 'mot1 mot2 mot3 ...'.
    Elle retourne un string, chaine de caractères, de réponse ainsi que le maximum de la 
    mesure de similarité.
    Le fonctionnement de cette fonction est identique à la fonction reponse(requete_utilisateur)
    sauf que le corpus utilisé est le corpus de questions issu du mode entrainement.
    '''

    '''etape 1'''
    requetes = paraphrase(
        requete_utilisateur)  # on considère que l'utilisateur émet une
    # requête ne contenant qu'une phrase
    # le corpus a été traité de la même manière que
    corpus = ' '.join(traitement_QU)
    # la requête : token°, stop_words, étiquetage, lemm°
    # IMPORTANT : on ne génère pas de paraphrases sur le corpus car cela générerait trop de possibles
    # c'est la que cela devient intéressant
    CORPUS = [corpus for k in requetes]

#    for k in range(len(requetes)) :
#        CORPUS[k] += ' '
#        CORPUS[k] += requetes[k][0]
    # print(CORPUS)
    CORPUS_TOKENS = []
    for i in CORPUS:
        CORPUS_TOKENS.append(tokenisation_phrase(i))

    '''etape 2'''
    IDF_Vect = TfidfVectorizer()  # transforme un doc en matrice TF-IDF
    IDF_liste = []
    for i in CORPUS_TOKENS:
        IDF = IDF_Vect.fit_transform(i)  # fit_transform(corpus)
        IDF_liste.append(IDF)
    # print(IDF_liste)
    ''' Cette partie là ne va plus du tout'''
    Coeff_sim_liste = [cosine_similarity(
        IDF_liste[i][-1], IDF_liste[i]) for i in range(len(IDF_liste))]  # WTF #FIXME
    # print(Coeff_sim_liste)
    '''etape 3'''
    Flat_liste = [Coeff_sim_liste[i].flatten()
                  for i in range(len(Coeff_sim_liste))]
    # Coeff_sim = array([[x,y,...]]) et la commande flatten permet d'obtenir array([x,y,z...])

#    for k in Flat_liste:
#        k.sort() #trie des différentes listes de la valeur la plus faible à la plus élevé #BUG
#        print(k)
    # print(Flat_liste)

    #Val_liste = [Flat_liste[k][2] for k in range(len(Flat_liste))]

    # retourne la valeur la plus élevée du coefficient de similarité #NON #BUG
    # Val_liste=Flat_liste[0].tolist()
    #Val_liste=[i.tolist() for i in Flat_liste]
    Val_liste = Flat_liste[:]
    print(Val_liste)
    '''etape 4'''
    print(requetes)
    Val_liste_pond = []
#    print(" de la variable Val_liste",len(Val_liste))
#    print(" la variable CORPUS_TOKENS",len(CORPUS_TOKENS))
    for k in range(len(Val_liste)):
        Val_liste_pond.append(Val_liste[k]*requetes[k][1])
    ''' premièrement il n'y pas de lien entre la longueur de requètes et la longueur de requêtes
     l'objectif est d'obtenir la ponderation finale entre les question du corpus et les synonyme trouvés
     len(Val_liste)== len (CORPUS_TOKENS)'''
# I beg your pardon
    '''etape 5'''
    print(Val_liste_pond)
    # N permet de déterminer quel élément de requête a la meilleur valeur de similarité
    # N = Val_liste_pond.index(max(Val_liste_pond))#OUI MAIS NON #FIXME
    N_val = 0
    for k in range(len(Val_liste_pond)):
        for i in range(len(Val_liste_pond[k])):
            if N_val <= Val_liste_pond[k][i]:
                N_val = Val_liste_pond[k][i]
                N = (k, i)
    print('Valliste', Val_liste)
    # n : indice de la valeur max de Val_liste
    #n = where(Coeff_sim_liste[N[0]][N[1]]== Val_liste[N[0]][N[1]])[1][0]
    """ Tuysse n==N """

    if(N_val == 0) or (N[1] > len(REP)):
        return ['Je suis désolée mais je n\'ai pas compris le sens de votre demande. Pouvez vous reformuler votre question ?', 0.5]  # HACK
    else:
        # on récupère la phrase et non la phrase traitée pour répondre à la question
        print("REP", REP)
        print("N1", N)
        return REP[N[1]], N_val


def TolBot():
    ''' Fonction permettant de chatter avec le bot.
    Elle ne prend pas de paramètre en entrée et permet de retourner une réponse de type 
    string à une requête que l'utilisateur rentrera dans la console (input()).
    Elle fonctionne à l'aide de balise. On initialise la balise comme étant True et tant 
    qu'elle le reste le programme continue.
    A la différene de chatbot(), cette fonction calcule 2 réponses (grâce aux 2 
    programmes de réponse) et on choisit la réponse la plus pertinente grâce à la
    mesure de similarité.    
    Cette fonction fait appel aux fonctions précédentes. 
    Rapportez vous aux doc-strings de ces fonctions pours plus d'information.
    '''

    signal = True
    print("TolBot : Bonjour, je m'appelle TolBot!")
    print("Je suis disponible pour répondre à tes questions sur le tolérancement.")
    print("N'hésite pas à me poser tes questions !")
    print("Pour partir, il te suffit de dire 'salut' !")

    while(signal == True):
        requete_utilisateur = input("Utilisateur : ")
        requete_utilisateur = requete_utilisateur.lower()
        if 'salut' not in requete_utilisateur:
            if remerciement(requete_utilisateur) != None:
                signal = False
                return("TolBot : {}".format(remerciement(requete_utilisateur)))
            else:
                if salut(requete_utilisateur) != None:
                    print("TolBot : {}".format(salut(requete_utilisateur)))
                else:
                    rep1, coeff1 = reponse_table(requete_utilisateur)[
                        0], reponse_table(requete_utilisateur)[1]
                    rep2, coeff2 = reponse(requete_utilisateur)[
                        0], reponse(requete_utilisateur)[1]
                    print("coeff1 , " + str(coeff1))
                    print("rep1 " + str(rep1))
                    print("coeff2 , " + str(coeff2))
                    print("coeff2 "+str(rep2))
                    if coeff1 >= coeff2:
                        print(
                            "TolBot : {} \n c'est une réponse de la table".format(rep1))
                        # stockage de la requête pour le transmettre à l'enseignant
                        aide_prof[requete_utilisateur] = rep1
                    else:
                        print(
                            "TolBot : {} \n c'est une réponse du corpus".format(rep2))
                        aide_prof[requete_utilisateur] = rep2
        else:
            signal = False
            return("TolBot : A la revoyure !")


# obtention du dictionnaire des synonymes
liste_noms = []

dic_syns = synonyme_liste(liste_noms)
# actuellement dic_syns est vide
# Donc tout les parties liées à la paraphrase n'ont pas de sens...
# traitement de la table issue du mode entrainement
QU, REP = repartition('table_QR')
QU_texte = ' '.join(QU)
traitement_QU = traitement(QU_texte.lower())
# Vectorizer=TfidfVectorizer(tokenizer=tokenisation_mot,norm=None,use_idf=False,smooth_idf=False)
# test=Vectorizer.fit_transform(corpus_traite)
liste_hier = extracted_data.test.Texte


def flatten(x):
    resultat = ""
    for item in x:
        if type(item) == list:
            resultat += flatten(item)
        else:
            resultat += item
    return resultat


doc = [flatten(i) for i in liste_hier]
sections_traité = [traitement(i) for i in doc]

from nltk.corpus import stopwords
import stanza
# import class_xml
import pickle
import re
nlp = stanza.Pipeline(processors="tokenize,pos,lemma",
                      lang="fr", logging_level="ERROR")

stop_words = stopwords.words("french")
ajout = ["être", "avoir", "bien", "ce", "cela", "celui", "cet", "cette", "ex",
         "à", "au", "en", "est", "il", "jamais", "qu'",
         "quel", "comment", "puis", "si", "la", "une", "en", "donc", "dire",
         "de", "ø", "ceci", "c\'", "pourtant", "cependant", "car"]
stop_words = stop_words+ajout

p = re.compile("\d\w|\w\d|\d[\W]*\d|\W|\d", re.IGNORECASE)

""" TOKENISATION """


def phrases(texte):
    """Séparer le texte brut en une liste de phrases """
    doc = nlp(texte)
    ans = []
    for sentence in doc.sentences:
        ans += [word.text for word in sentence.words
                if word.text not in stop_words and not p.match(word.text) and word.text != "."]
    return ans


def token(texte):
    """ sépare le texte brute en tokens, renvoie la liste des tokens"""
    doc = nlp(texte)
    tokens = []
    for phrase in doc.sentences:
        for word in phrase.words:
            if word.text not in stop_words and not p.match(word.text):
                tokens.append(str(word.text))
    return tokens


"""LEMMATISATION"""


def lemma(texte):
    """prend un texte un entrée et renvoie la liste des lemmes """
    # texte = texte.lower()
    doc = nlp(texte)
    answer = []
    for phrases in doc.sentences:
        for word in phrases.words:
            if word.lemma not in stop_words and not p.match(word.lemma):
                # print(word.lemma)
                answer.append(word.lemma)
    return answer


class Texte():
    Ensemble_txt = []
    # __voc__={}
    __voc__ = pickle.load(open("dico_mot_index.p", 'rb'))
    # ce dico vient du traitement du corpus
    # voir le fichier corpus.py

    def __init__(self, ficelle: str):
        self.txt_brut = ficelle.strip()
        self.txt_brut = self.txt_brut.lower()
        self.txt_token = token(self.txt_brut)
        self.txt_lemma = lemma(self.txt_brut)
        self.txt_lemma_str = " ".join(self.txt_lemma)
        self.oh = []
        Texte.Ensemble_txt.append(self)

    def __one_hot__(self):
        assert (len(self.__voc__) >
                0), "Vocabulary error: please use the vocabulary()"
        """ on suppose que le vocabulaire est ok """
        self.oh = [self.__voc__[i] for i in self.txt_lemma]

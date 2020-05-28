# -*- coding: utf-8 -*-
import pickle
import class_xml
import class_texte


def Traitement(document):
    assert(isinstance(document, class_xml.document))
    tree = document.treelib
    liste_tout = tree.all_nodes()
    liste_inter = []
    liste_finale = []
    for i in liste_tout:
        if i.tag == "image" and len(i.data.texte) != 0:
            liste_inter.append(i)
        elif i.tag in ["liste", "image"]:
            continue
        else:
            liste_inter.append(i)

    for item in liste_inter:
        ID = (item.identifier, document.id)
        if isinstance(item.data, class_xml.section):
            if len(item.data.Nom) > 0 and item.data.Nom is not None:
                liste_finale.append(
                    (class_texte.Texte(item.data.Nom), "Section", ID))
        elif isinstance(item.data, class_xml.document):
            # print(item.data.Titre)
            if len(item.data.Titre) > 0 and item.data.Titre is not None:
                liste_finale.append(
                    (class_texte.Texte(item.data.Titre), "Doc", ID))
        else:
            if len(item.data.texte.strip()) > 0 and item.data.texte.strip() is not None:
                # print("data :"+item.data.texte)
                liste_finale.append(
                    (class_texte.Texte(item.data.texte), "Autre", ID))
    return liste_finale


def vocabulary(doc_traites):
    """dico avec une correspondance
    Clé entier int et valeurs string token normalisé
    dico r réalise la correspondance inverse
    Ce type de codage est très naifs pas de Hashmap
    """
    dico = {}
    dico_r = {}
    liste_tokens = []
    for doc in doc_traites:
        for (object_texte, nature, ID) in doc:
            for token in object_texte.txt_lemma:
                liste_tokens.append(token)
    liste_unique = list(set(liste_tokens))
    liste_unique.insert(0, "RESERVED")
    # print(liste_unique[0:5])
    for indice, valeur in enumerate(liste_unique):
        dico[indice] = valeur
        dico_r[valeur] = indice
    return dico, dico_r


def Traitement_Corpus(corpus):
    """ le corpus est une liste de fichier XML """
    liste_doc = []
    # création des documents
    for fichier in corpus:
        liste_doc.append(class_xml.document(fichier))
    # Traitement de documents
    doc_traites = []
    for doc in liste_doc:
        doc_traites.append(Traitement(doc))
    id2mot, mot2id = vocabulary(doc_traites)
    return doc_traites, liste_doc


Corpus = ["/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/1_spécif ISO_Bases juin 2017 v5 ANSELMETTI/doc1.xml",
          "/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/2_Spécif ISO_Complément 2017 v4ANSELMETTI/essai.xml",
          "/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/E_GPS_2_Bases de la cotation v11/EGPS2.xml",
          "/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/E_GPS_3_Spécifications complémentaires v9/EGPS3.xml"]
liste_docments_traités, liste_doc_xml = Traitement_Corpus(Corpus)
test_dico, test_2 = vocabulary(liste_docments_traités)
input_liste = [object_texte.txt_lemma_str for doc in liste_docments_traités for (
    object_texte, nature, ID) in doc]
liste_id = [ID for doc in liste_docments_traités for (
    object_texte, nature, ID) in doc]
class_texte.Texte.__voc__ = test_2
print("taille du vocabulaire", len(test_2))
pickle.dump(test_2, open("dico_mot_index.p", 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(liste_docments_traités, open("docment_traites.p", 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(liste_doc_xml, open("liste_doc_xml.p", 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(liste_ID for ID in Doc)
"""Modèle du corpus """

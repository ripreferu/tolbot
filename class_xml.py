import pathlib as pl
import treelib as tl
import random
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
#tree=ET.parse("/home/pierre/Téléchargements/Corpus_BA/1_spécif ISO_Bases juin 2017 v5 ANSELMETTI/essai_tei.xml")
tree=ET.parse("/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/2_Spécif ISO_Complément 2017 v4ANSELMETTI/2_Spécif ISO_Complément 2017 v4ANSELMETTI.xml")
root=tree.getroot()
doc=root[1][0]
ET.register_namespace("",'{http:\\www.tei-c.org/ns/1.0}')
ns='{http://www.tei-c.org/ns/1.0}'
# for child in document.iter():
#     print(child.tag)
###
# root[1] ==> body
# root[1][0] ==> interessant

# def traitement_titre(Liste):
#     for k in Liste:
#         if k[0].tag=="head":
#             print("le premier élément à K est bien un head")
#         else:
#             print("mon hypothèse est fausse")
#             ##### Traitement
#test=section_parser(document)
#nouv=[section_parser(i) for i in test]
# descend de 1 étage
#for i in nouv:
#    traitement_titre(i)
# len Liste == 8 cad le nombre de sections de documents en question

def recur(liste,reponse=[],n=0,option=0,**kwargs):
    if n==0:
        reponse=[(n,len(liste))]
        loc=[]
        n+=1
    if option==0:
        for index,item in enumerate(liste):
            if type(item)==list:
                m=len(item[:])
                prev_index=kwargs.get("pindex",None)
                if prev_index is not None:
                    reponse.append((n,m,prev_index)[:])
                    loc=prev_index.append(index)
                else:
                    reponse.append((n,m,[index])[:])
                    loc=[index]
                n+=1
                recur(item,reponse,n=n,pindex=loc)
                n-=1
    else:
        for item in liste:
            if type(item)==list:
                m=len(item[:])
                reponse.append((n,m)[:])
                n+=1
                recur(item,reponse,n=n,option=option)
                n-=1
        # else:
        #     print("⛩")
        #     reponse.append((n,len(liste))[:])
    return reponse[:]
def dimension(liste):
    res=recur(liste,option=1)[:]
    res=set(res)
    unique={}
    for depth,longueur in res:
        if depth not in unique:
            unique[depth]=longueur
        elif longueur>unique[depth]:
            unique[depth]=longueur
    reponse=[unique[i] for i in range(len(unique))]
    return tuple(reponse)
### Il est plus simple de récupérer le titre de chaque section depuis le div id
def transformation(longueur_voulue,liste):
    liste=np.pad(liste,(0,longueur_voulue-len(liste)),mode="constant",constant_values=np.nan)
    return liste
def call_fonction(liste_ind,liste_par):
    rep=liste_par
    for k in liste_ind:
        rep=rep[k]
    return rep

def creer_array(liste):
    # rep = liste
    struc = recur(liste)[1:]
    dim = dimension(liste)
    for i in struc:
        prof_act=i[0]
        longueur_act=i[1]
        indice=i[2]
        liste_de_travail=call_fonction(indice,liste)
        longueur_souhait=dim[prof_act]
        liste_de_travail=np.array(transformation(longueur_souhait,liste_de_travail))
        print(liste_de_travail)
        print(indice)
def flatten(x):
    resultat=""
    for item in x:
        if type(item)==list:
            resultat+=flatten(item)
        else:
            resultat+=item
    return resultat

class document():
    """
    self.tree est l'arbre XML
    self.racine_doc est la racine de l'arborescence XML
    self.Metadata est l'arborescence XML
    self.document est une arborescence XML
    self.Titre est  le titre sous format string
    self.Auteur est l'auteur sous forme d'un string
    self.sections est la liste des objets sections
    self.Texte est le texte contenu dans ces sections
    """
    def __init__(self,chemin):
        self.id=random.random()
        self.chemin=pl.Path(chemin)
        self.tree=ET.parse(chemin)
        self.racine_doc= self.tree.getroot()
        ET.register_namespace("xml",'http://www.tei-c.org/ns/1.0')
        self.dico=self.racine_doc.tag[:-3]
        # en supposant que le fichier a bien été formater sous la format XML TEI
        self.document = self.racine_doc[1][0]
        self.metadata = self.racine_doc[0][0]
        #### informations relatives à l'oeuvre self.metadata [0]
        #### informations relatives à la publication self.metadata[1]
        ### informations relatives à la source
        self.treelib=tl.Tree()
        self.Titre=None
        for children in self.metadata.iter():
            if children.tag[-5:]=="title":
                self.Titre=children.text
            elif children.tag[-6:]=="author":
                self.Auteur=children.text
        
        self.sections=[]
        self.definir_les_sections()
        self.Texte=[]
        #self.texte_documents()
        self.treelib.create_node("Document ",self.id, data=self)
        self.creation_arbre()
        # dim=dimension(self.Texte) 
        # profondeur=len(dim)
        
        # rv=list(zip(*self.Texte))
        # self.D={"chemin":self.chemin, "titre":self.Titre ,}
        # liste_test= [*self.sections, [rv[i][:] for i in range(len(rv))]]
        # self.data=pd.DataFrame(data=self.D)
            
    def ajout_section_arbre(self,section_add):
        assert isinstance(section_add,section) or isinstance(section_add,liste) or isinstance(section_add,paragraph)
        
        for children in section_add.Contenu:
            if isinstance(children,section):
                self.treelib.create_node(tag=str(children), identifier=children.id, parent=section_add.id, data=children)
                self.ajout_section_arbre(children)
            elif isinstance(children,liste):
                self.treelib.create_node(tag=str(children),identifier=children.id,parent=section_add.id, data=children)
                self.ajout_section_arbre(children)
            elif isinstance(children,paragraph):
                if children.texte!=' None ':
                    self.treelib.create_node(tag=str(children),identifier=children.id,parent=section_add.id,data=children)
                if len(children.Contenu)>0:
                    for item in children.Contenu:
                        if isinstance(item,figure) or isinstance(item,image):
                            item.url=self.chemin.parents[0].joinpath(item.url)
                            self.treelib.create_node(tag=str(item),identifier=item.id,parent=section_add.id,data=item)
                            #data=self.chemin.parents[0]+children.url)
                        if isinstance(item,note):
                            self.treelib.create_node(tag=str(item),identifier=item.id,parent=section_add.id,data=item)
            else:
                self.treelib.create_node(tag=str(children),identifier=children.id,parent=section_add.id, data=children)
    def creation_arbre(self):
        for section in self.sections:
            self.treelib.create_node(tag=str(section),identifier=section.id,parent=self.id,data=section)
            # for contenu in section.Contenu:
            #     self.treelib.create_node(tag=str(contenu),identifier=contenu.id,parent=section.Titre,data=contenu.texte)
            self.ajout_section_arbre(section)

    def definir_les_sections(self):
        Liste=[]
        for child in self.document:
            if child.tag==self.dico+"div":
                Liste.append(child)
        for i in  Liste:
            item=section(i)
            self.sections.append(item)
    def texte_documents(self):
        Tree=self.treelib
        node_sec_l=Tree.children(Tree.root) #liste de noeuds
        liste_texte=[]
        for node_sec in node_sec_l:
            ensemble_sous_section=[Tree.get_node(ide)
                                       for ide in Tree.expand_tree(node_sec.data.id,filter=lambda x: \
                                                                   x.tag not in ["liste","paragraphe","image","note"])]
            liste_titre=[node.data.Titre for node in ensemble_sous_section]
            liste_inter=[flatten(node.data.texte) for node in Tree.leaves(node_sec.data.id) if node.tag!="image"]
            ajout_texte=" ".join(liste_inter)
            if len(liste_titre)>0:
                ajout_titre=" ".join(liste_titre[:])
                ajout_texte+=ajout_titre
            liste_texte.append(ajout_texte[:])
        return liste_texte
    def Forme_travail_texte(self):
        """ renvoie la liste utile pour la suite du traitement nlp
        sous la forme d'une liste de couple:
        '(Titre, "Doc")'
        '(Nom de la section,"Section")'
        '(Contenu autre, "Autre")'
        """
        pass

class liste():
    def __init__(self,element_xml):
        self.id=random.random()
        self.Contenu=[]
        self.texte=""
        self.pt_mnt_xml=element_xml
        self.definir_Contenu()
    def __repr__(self):
        return "liste"
    def definir_Contenu(self):
        Sous_contenu_xml=[]
        for itemxml in self.pt_mnt_xml:
            for foo in itemxml:
                Sous_contenu_xml.append(foo)
        #Sous_contenu_xml=[item[:] for item in self.pt_mnt_xml[:]]
        #print(Sous_contenu_xml)
        for itemxml in Sous_contenu_xml:
            if itemxml.tag[-3:]=="div":
                nouv_section=section(itemxml)
                self.Contenu.append(nouv_section)
                self.texte+=nouv_section.texte_sect()
            elif itemxml.tag[-6:]=="figure":
                self.Contenu.append(image(itemxml)) #à coder
            elif itemxml.tag[-1:]=="p":
                nouv_para=paragraph(itemxml)
                self.Contenu.append(nouv_para)
                self.texte+=nouv_para.texte.strip()
            elif itemxml.tag[-4:]=="list":
                sous_liste=liste(itemxml)
                self.Contenu.append(sous_liste)
                self.texte+=sous_liste.texte.strip()
class section():
    def __init__(self,element):
        assert element.tag[-3:]=="div"
        ID_attrib=list(element.attrib.keys())[-1]
        variable_id=element.attrib[ID_attrib]
        variable_id = variable_id.replace("-"," ")
        self.Titre=variable_id
        self.Nom=element[0].text
        ### definir le contenu d'une section
        self.texte=[]
        self.Contenu=[]
        # self.matrice=np.array()
        Contenu_xml= element[:]
        for souscontenu in Contenu_xml:
            if souscontenu.tag[-3:]=="div":
                self.Contenu.append(section(souscontenu))
            elif souscontenu.tag[-6:]=="figure":
                self.Contenu.append(image(souscontenu))
            elif souscontenu.tag[-1:]=="p":
                self.Contenu.append(paragraph(souscontenu))
            elif souscontenu.tag[-4:]=="list":
                self.Contenu.append(liste(souscontenu))
            elif souscontenu.tag[-5:]=="quote":
                #print(souscontenu[:])
                self.Contenu.append(paragraph(souscontenu[0]))
        self.texte_sect()
        self.id=random.random()
    def __repr__(self):
        return str(self.Nom)
    def __getitem__(self,index):
        if index==slice:
            return self.Contenu[index.start,index.stop]
        else:
            return self.Contenu[index]
    def texte_sect(self):
        """ génère le texte des sections """
        texte=[]
        for item in self.Contenu:
            if isinstance(item,section):
                item.texte_sect()
                self.texte.append(item.texte)
            else:
                soustexte=item.texte
                soustexte=soustexte.strip()
                #print(soustexte)
                if len(soustexte)>0 and soustexte!='None':
                    #soustexte=soustexte.strip()
                    #soustexte=soustexte.strip("\\n")
                    texte.append(soustexte)
                    #print(texte)
                self.texte=texte
        #for item in self.Contenu:
                #print(type(item.texte))
        #return self.texte
    def matrice(self):
        m=len(self.Texte)
        n=max([ i for i in self.Texte])
class paragraph():
    def __init__(self,element):
        #assert element.tag[-3:]!='head'
        # paragraph doit être instancié à partir des sections
        #print(element[:])
        self.id=random.random()
        self.Contenu=[]
        self.texte=""
        if type(element.text)==str:
            Texte_a_strip=element.text
            Texte_a_strip=Texte_a_strip.replace("\\n","\n")
            Texte_a_strip=Texte_a_strip.replace("\n","")
            if len(Texte_a_strip)>0 and Texte_a_strip!="None":
                self.texte+=Texte_a_strip.replace("\n","")
            #self.texte+=element.text.strip()
        for children in element:
            #print(children)
            #print(children.tag[-3:])
            if children.tag[-6:]=="figure":
                # if len(children.attrib)!=0:
            #         print(children.attrib)
            #     print(children[0].tag[-3:])
                self.Contenu.append(image(children))
                if str(children.tail) is not None:
                    self.texte+=" "+str(children.tail).replace("\n","")
                #possiblité d'ajouté une image
            elif children.tag[-4:]=="note":
                self.Contenu.append(note(children))
                if str(children.tail) is not None:
                    self.texte+=" "+str(children.tail).replace("\n","")
            elif children.tag[-2:]=="lb" and children.tail is not None:
                self.texte+=" "+str(children.tail).strip()
        self.texte+=" "
        #self.XML=element
        ######## self.texte=" ".join(element.itertext())
        #self.texte=element.find("p").text
        #self.texte=self.texte.replace('\n',' ')
        #self.texte=self.texte.strip()
    def __repr__(self):
        return self.texte
class figure():
    def __init__(self,element):
        self.id=random.random()
        if element.findall(ns+"graphic"):
            self=image(element)
        elif element.findall(ns+"formula"):
            self=equation(element)
    def __repr__(self):
        return "figure"
class image(figure):
    def __init__(self,element):
        self.id=random.random()
        # print(element[:])
        nom=ns+'graphic'
        graphic=element.find(nom)
        #print(graphic)
        self.texte=""
        # print(element[0].attrib)
        self.url=graphic.attrib["url"]
        #print('ok')
        if element.findall(ns+'head'):
            titre=element.find(ns+'head')
            #print(titre.text)
            self.texte=titre.text
        if element.findall(ns+'figDesc'):
            figDesc=element.find(ns+'figDesc')
            #print(figDesc.text)
    def __repr__(self):
        if len(self.texte)!=0:
            return self.texte
        else:
            return "image"

# class equation(figure):
#     def init(self,element):
#         assert element.tag=="figure"
#         assert element.attrib["type"]=="math"
#         formula=element.find("formula")
#         self.notation=formula.attrib["notation"]
#         self.contenu=formula.text

class note():
    def __init__(self,element):
        self.id=random.random()
        self.texte=element.text
        #print(self.texte)
    def __repr__(self):
        return "note"
#test=document("/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/2_Spécif ISO_Complément 2017 v4ANSELMETTI/2_Spécif ISO_Complément 2017 v4ANSELMETTI.xml")
# test=document("/home/pierre/Documents/CorpusV2/Corpus_augmente/Docx/2_Spécif ISO_Complément 2017 v4ANSELMETTI/essai.xml")
# #test.definir_les_sections()
# #problème=test.document[5][1]
# #titi=liste(test.document[4][1][1][2])
# tree=test.treelib
# liste_para=[node.data.texte for node in tree.filter_nodes(lambda x: x.tag == "paragraphe")]


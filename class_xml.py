import pathlib as pl
import treelib as tl
import random
import xml.etree.ElementTree as ET

ns = '{http://www.tei-c.org/ns/1.0}'


def flatten(x):
    resultat = ""
    for item in x:
        if type(item) == list:
            resultat += flatten(item)
        else:
            resultat += item
    return resultat


class document():
    """
    Cette classe interprète un document XML selon la convention TEI
    et le transforme en objet pythons dont les classes sont détaillées
    si après.
    Voici qu'elle attribut d'un objet document
    self.tree est l'arbre XML
    self.racine_doc est la racine de l'arborescence XML
    self.Metadata est l'arborescence XML
    self.document est une arborescence XML
    self.Titre est  le titre sous format string
    self.Auteur est l'auteur sous forme d'un string
    self.sections est la liste des objets sections
    self.Texte est le texte contenu dans ces sections
    """

    def __init__(self, chemin):
        '''pour creer un objet document il suffit de donner le chemin où path
        vers le fichier XML au format tei
        L'objet est complet dès sa création. les objet internes sont recréer
        dès l'intanciation'''
        self.id = random.random()
        self.chemin = pl.Path(chemin)
        self.tree = ET.parse(chemin)
        self.racine_doc = self.tree.getroot()
        self.dico = self.racine_doc.tag[:-3]
        # en supposant que le fichier a bien été formater sous la format
        # xml/tei
        self.document = self.racine_doc[1][0]
        self.metadata = self.racine_doc[0][0]
        # informations relatives à l'oeuvre self.metadata [0]
        # informations relatives à la publication self.metadata[1]
        # informations relatives à la source
        self.treelib = tl.Tree()
        self.Titre = None
        for children in self.metadata.iter():
            if children.tag[-5:] == "title":
                self.Titre = children.text
            elif children.tag[-6:] == "author":
                self.Auteur = children.text

        self.sections = []
        self.definir_les_sections()
        self.Texte = []
        self.treelib.create_node("Document ", self.id, data=self)
        self.creation_arbre()

    def ajout_section_arbre(self, section_add):
        '''Cette fonction ajoute recursivement les sections et
        leurs contenu l'arbre nécessite probablement une refactorisation
        en créant un fonction générique pour chaque objet
        ---- WARNING ----
        This code is messy but works
        ----------------
        '''
        assert isinstance(section_add, section) or isinstance(
            section_add, liste) or isinstance(section_add, paragraph)
        for children in section_add.Contenu:
            if isinstance(children, section):
                self.treelib.create_node(
                    tag=str(children), identifier=children.id,
                    parent=section_add.id, data=children)
                self.ajout_section_arbre(children)
            elif isinstance(children, liste):
                self.treelib.create_node(
                    tag=str(children), identifier=children.id,
                    parent=section_add.id, data=children)
                self.ajout_section_arbre(children)
            elif isinstance(children, paragraph):
                if children.texte != ' None ':
                    self.treelib.create_node(
                        tag=str(children), identifier=children.id,
                        parent=section_add.id, data=children)
                if len(children.Contenu) > 0:
                    for item in children.Contenu:
                        if isinstance(item, figure) or isinstance(item, image):
                            item.url = self.chemin.parents[0].joinpath(
                                item.url)
                            self.treelib.create_node(
                                tag=str(item), identifier=item.id,
                                parent=section_add.id, data=item)
                        if isinstance(item, note):
                            self.treelib.create_node(
                                tag=str(item), identifier=item.id,
                                parent=section_add.id, data=item)
            else:
                self.treelib.create_node(
                    tag=str(children), identifier=children.id,
                    parent=section_add.id, data=children)

    def creation_arbre(self):
        '''Créé l'arbre -*Treelib*- du document
        Cette fonction est appélé dès l'instanciation de l'object document'''
        for section in self.sections:
            self.treelib.create_node(
                tag=str(section), identifier=section.id, parent=self.id,
                data=section)
            self.ajout_section_arbre(section)

    def definir_les_sections(self):
        Liste = []
        for child in self.document:
            if child.tag == self.dico+"div":
                Liste.append(child)
        for i in Liste:
            item = section(i)
            self.sections.append(item)

    def texte_documents(self):
        ''' Actuellement non utilisé
        Cette fonction est censé extraire les données textuelle de l'arbre
        Cette fonctionnalité est réalisé dans la classe'''
        Tree = self.treelib
        node_sec_l = Tree.children(Tree.root)  # liste de noeuds
        liste_texte = []
        for node_sec in node_sec_l:
            ensemble_sous_section = [
                Tree.get_node(ide) for ide in Tree.expand_tree(
                    node_sec.data.id, filter=lambda x:x.tag not in
                    ["liste", "paragraphe", "image", "note"])]
            liste_titre = [node.data.Titre for node in ensemble_sous_section]
            liste_inter = [flatten(node.data.texte) for node in Tree.leaves(
                node_sec.data.id) if node.tag != "image"]
            ajout_texte = " ".join(liste_inter)
            if len(liste_titre) > 0:
                ajout_titre = " ".join(liste_titre[:])
                ajout_texte += ajout_titre
            liste_texte.append(ajout_texte[:])
        return liste_texte


class liste():
    def __init__(self, element_xml):
        self.id = random.random()
        self.Contenu = []
        self.texte = ""
        self.pt_mnt_xml = element_xml
        self.definir_Contenu()

    def __repr__(self):
        return "liste"

    def definir_Contenu(self):
        Sous_contenu_xml = []
        for itemxml in self.pt_mnt_xml:
            for foo in itemxml:
                Sous_contenu_xml.append(foo)
        for itemxml in Sous_contenu_xml:
            if itemxml.tag[-3:] == "div":
                nouv_section = section(itemxml)
                self.Contenu.append(nouv_section)
                self.texte += nouv_section.texte_sect()
            elif itemxml.tag[-6:] == "figure":
                self.Contenu.append(image(itemxml))  # à coder
            elif itemxml.tag[-1:] == "p":
                nouv_para = paragraph(itemxml)
                self.Contenu.append(nouv_para)
                self.texte += nouv_para.texte.strip()
            elif itemxml.tag[-4:] == "list":
                sous_liste = liste(itemxml)
                self.Contenu.append(sous_liste)
                self.texte += sous_liste.texte.strip()


class section():
    def __init__(self, element):
        assert element.tag == ns+"div"
        ID_attrib = list(element.attrib.keys())[-1]
        variable_id = element.attrib[ID_attrib]
        variable_id = variable_id.replace("-", " ")
        self.Titre = variable_id
        self.Nom = element[0].text
        # definir le contenu d'une section
        self.texte = []
        self.Contenu = []
        # self.matrice=np.array()
        Contenu_xml = element[:]
        for souscontenu in Contenu_xml:
            if souscontenu.tag == ns+"div":
                self.Contenu.append(section(souscontenu))
            elif souscontenu.tag == ns+"figure":
                self.Contenu.append(image(souscontenu))
            elif souscontenu.tag == ns+"p":
                self.Contenu.append(paragraph(souscontenu))
            elif souscontenu.tag == ns+"list":
                self.Contenu.append(liste(souscontenu))
            elif souscontenu.tag == ns+"quote":
                self.Contenu.append(paragraph(souscontenu[0]))
        self.texte_sect()
        self.id = random.random()

    def __repr__(self):
        return str(self.Nom)

    def __getitem__(self, index):
        if index == slice:
            return self.Contenu[index.start, index.stop]
        else:
            return self.Contenu[index]

    def texte_sect(self):
        """ génère le texte des sections """
        texte = []
        for item in self.Contenu:
            if isinstance(item, section):
                item.texte_sect()
                self.texte.append(item.texte)
            else:
                soustexte = item.texte
                soustexte = soustexte.strip()
                if len(soustexte) > 0 and soustexte != 'None':
                    texte.append(soustexte)
                self.texte = texte


class paragraph():
    def __init__(self, element):
        '''cette fonction créé un paragraphe à partir d'un objet xml '''
        self.id = random.random()
        self.Contenu = []
        self.texte = ""
        if type(element.text) == str:
            Texte_a_strip = element.text
            Texte_a_strip = Texte_a_strip.replace("\\n", "\n")
            Texte_a_strip = Texte_a_strip.replace("\n", "")
            if len(Texte_a_strip) > 0 and Texte_a_strip != "None":
                self.texte += Texte_a_strip.replace("\n", "")
        for children in element:
            if children.tag == ns+"figure":
                self.Contenu.append(image(children))
                if str(children.tail) is not None:
                    self.texte += " "+str(children.tail).replace("\n", "")
                # possiblité d'ajouté une image
            elif children.tag == ns+"note":
                self.Contenu.append(note(children))
                if str(children.tail) is not None:
                    self.texte += " "+str(children.tail).replace("\n", "")
            elif children.tag == ns+"lb" and children.tail is not None:
                self.texte += " "+str(children.tail).strip()
        self.texte += " "

    def __repr__(self):
        return self.texte


class figure():
    def __init__(self, element):
        '''L'objet figure est instancié depuis un élement xml '''
        self.id = random.random()
        if element.findall(ns+"graphic"):
            self = image(element)
        # elif element.findall(ns+"formula"):
        #     self = equation(element)

    def __repr__(self):
        return "figure"


class image(figure):
    def __init__(self, element):
        self.id = random.random()
        nom = ns+'graphic'
        graphic = element.find(nom)
        self.texte = ""
        self.url = graphic.attrib["url"]
        if element.findall(ns+'head'):
            titre = element.find(ns+'head')
            self.texte = titre.text
        if element.findall(ns+'figDesc'):
            figDesc = element.find(ns+'figDesc')
            self.texte += figDesc.text  # inutilisé actuellement

    def __repr__(self):
        if len(self.texte) != 0:
            return self.texte
        else:
            return "image"


class equation(figure):
    ''' not implemented yet'''
    pass


class note():
    def __init__(self, element):
        self.id = random.random()
        self.texte = element.text

    def __repr__(self):
        return "note"

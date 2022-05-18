import os.path
import mangoes.utils
from bs4 import BeautifulSoup

class MyXMLParser:

    def __init__(self, source):
        self.ignored = ['lb','pb']
        self.source = source
        with open(source,encoding='utf-8') as f:
            self.soup = BeautifulSoup(f,features="xml")
        
    def parse_sentences(self):
        self.sentences = self.get_sentences(self.soup,self.ignored)
        return self.sentences

    def _sentence_containers(self,node):
        containing_wf = set()
        words = node.find_all('wf')
        for wf in words:
            containing_wf = containing_wf | {wf.findParent().name}
        return containing_wf

    def _expand_sentence(self,element,current_sentence,sentences,should_end: bool, ignored):
        if element.name == 'hi':
            #This is an highlighted section of text
            children = element.find_all(recursive=False)
            for child in children:
                should_end = self._expand_sentence(child,current_sentence,sentences, should_end, ignored)
        elif element.name == 'wf':
            if should_end:
                if element.get('pos')!='PONCT':
                    #It the begining of a new sentence, the word begins with a capital letter.
                    sentences.append( current_sentence.copy() )
                    current_sentence.clear()
                    should_end = False
                else:
                    #False alarm, it is probably just an abbreviation ("Dr.")
                    should_end = False
            #This is a word in the current sentence
            current_sentence.append(element)
            if element.get('lemma') in '.?!")]}»—―”':
                should_end = True
        elif current_sentence and element.name not in ignored:
            #Ignore linebreak
            #If a sentence was being written, it is interrupted.
            should_end = True
        return should_end

    def get_sentences(self,soup,ignored_tags):
        containers = self._sentence_containers(soup)
        sentences = list()
        should_end = False
        
        for container_tag in containers:
            if container_tag != 'hi':
                nodes = soup.find_all(container_tag)
                for node in nodes:
                    wf_children = node.find_all('wf',recursive=False)
                    if wf_children:
                        all_children = node.find_all(recursive=False)
                        current_sentence = list()
                        for child in all_children:
                            should_end = self._expand_sentence(child,current_sentence,sentences,should_end,ignored_tags)
        return sentences
        
        


class MyXmlSentenceGenerator(mangoes.utils.reader.AnnotatedSentenceGenerator):
    """Sentence generator for an XML source

    See Also
    --------
    :class:`SentenceGenerator`
    """
    PUNCTUATION_TAG = 'PONCT'

    def __init__(self, source, lower=False, digit=False, ignore_punctuation=False, only_lemma =False):
        super().__init__(source, lower, digit, ignore_punctuation)
        if os.path.exists(self.source):
            self.sentences = self._sentences_from_files
        else:
            self.sentences = self._sentences_from_string
    

    @classmethod
    def _remove_punctuation(cls, sentence):
        return [t for t in sentence if t.POS != cls.PUNCTUATION_TAG]


    def _parse_token(self, xml_token):
        return self.Token(form=xml_token.get('word'),
                          POS=xml_token.get('pos'),
                          lemma=xml_token.get('lemma'))

    def _sentences_from_xml(self, xml_sentences):
        for xml_sentence in xml_sentences:
            yield self.transform([self._parse_token(token) for token in xml_sentence])
        return

    def _sentences_from_files(self):
        for xml_file in mangoes.utils.io.recursive_list_files(self.source):
            xmlparser = MyXMLParser(xml_file)
            yield from self._sentences_from_xml(xmlparser.parse_sentences())

    def _sentences_from_string(self):
        xmlparser = MyXMLParser(self.source)
        yield from self._sentences_from_xml(xmlparser.parse_sentences())


class MyXmlSentenceGenerator_LEMMA(MyXmlSentenceGenerator):
    def _parse_token(self, xml_token):
        return self.Token(form=xml_token.get('lemma'),
                          POS=xml_token.get('pos'),
                          lemma=xml_token.get('lemma'))

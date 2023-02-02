from params import SELECTED_TAGS

NOUNS = { 'nd1', 'nn', 'nn1', 'nn2', 'nna', 'nnb', 'nnl1', 'nnl2', 'nno', 'nno2', 'nnt1', 'nnt2', 'nnu', 'nnu1', 'nnu2' }
PROPER_NOUNS = {'np', 'np1', 'np2', 'npd1', 'npd2', 'npm1', 'npm2'}
ARTICLES = {'at','at1'}
CONJUNCTIONS = {'cc','ccb','cs','csa','csn','cst','csw'}
DETERMINERS = {'da','da1','da2','dar','dat','db','db2','dd','dd1','ddq','ddqge','ddqv'}
PREPOSITIONS = {'if','ii','io','iw'}
NUMBERS = {'mc','mc1','mc2','mcge','mcmc','md','mf'}
PRONOUNS = {'pn','pn1','pnqo','pnqs','pnqv','pnx1','ppge','pph1','ppho1', 'ppho2', 'pphs1', 'pphs2', 'ppio1', 'ppio2', 'ppis1', 'ppis2', 'ppx1', 'ppx2', 'ppy'}
ADVERBS = {'ra','rex','rg','rgq','rgqv', 'rgr', 'rgt', 'rl', 'rp', 'rpk', 'rr', 'rrq', 'rrqv', 'rrr', 'rrt', 'rt'}
ADJECTIVES = { 'jj', 'jjr', 'jjt', 'jk' }
# Lexical verbs only, modals are excluded
VERBS = { 'vv0', 'vvd', 'vvg', 'vvgk' , 'vvi', 'vvn', 'vvnk' , 'vvz'}

MAPPER = dict([
    ('N',NOUNS), 
    ('PN',PROPER_NOUNS),
    ('ART',ARTICLES),
    ('CONJ',CONJUNCTIONS),
    ('DET',DETERMINERS),
    ('PRE',PREPOSITIONS),
    ('NB',NUMBERS),
    ('PRO',PRONOUNS),
    ('ADV',ADVERBS),
    ('ADJ',ADJECTIVES),
    ('V',VERBS)
])

selected_lists = [MAPPER[tag] for tag in SELECTED_TAGS]

def replace_posTag(pos_tag):
    # common nouns only, no proper nouns
    for pos_id, pos_list in zip(SELECTED_TAGS,selected_lists):
        if type(pos_tag) != str:
            return None
        if pos_tag.lower() in pos_list :
            return pos_id
    return None

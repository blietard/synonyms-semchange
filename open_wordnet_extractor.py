import pickle
import argparse
import pandas as pd
from collections import defaultdict
from params import OPENWORDNET_DB_FOLDER, WORDS_FOLDER

parser = argparse.ArgumentParser(description='Read OpenWordNet database.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
args = parser.parse_args()

synsets = defaultdict(list)
words_and_senses = list()

pos = {'ADJ' : 'adj', 'N':'noun', 'V':'verb'}[args.pos]

with open(OPENWORDNET_DB_FOLDER + 'index.' + pos, 'r', encoding='utf-8') as f:
    nb_line = 0
    while True:
        line = f.readline()
        if line == '':
            break
        nb_line += 1
        if line[0] != ' ':
            if nb_line%1000 == 0:
                print(nb_line,end='\r')
            l = line.split()
            lemma = l.pop(0)
            l.pop(0) # pos tag already known
            sense_cnt = int(l.pop(0))  #nb of synsets
            ptr_cnt = int(l.pop(0))
            for i in range(ptr_cnt):
                l.pop(0) # Ignore pointers
            l.pop(0) # nb of synsets (redundant)
            l.pop(0) # ignore nb of ranked synsets
            for i in range(sense_cnt):
                synsets[l.pop(0)].append(lemma)
            words_and_senses.append( [lemma,sense_cnt] )

words_and_synonyms = defaultdict(set)
for synset in synsets.values():
    for word in synset:
        words_and_synonyms[word] |= set(synset)

nb_syns_per_word = list()
words = list()
for word, syns in words_and_synonyms.items():
    words.append(word)
    nb_syns_per_word.append(len(syns)-1)

df = pd.DataFrame(words_and_senses,columns=['words','nb_senses'])
df.set_index('words',inplace=True)
df['nb_syns'] = pd.Series(data=nb_syns_per_word,index=words)

with open(f'{WORDS_FOLDER}/openwordnet/synsets_by_ID_{args.pos}.pkl','wb') as pkl_f:
    pickle.dump(obj=synsets, file=pkl_f)
with open(f'{WORDS_FOLDER}/openwordnet/synsets_by_words_{args.pos}.pkl','wb') as pkl_f:
    pickle.dump(obj=words_and_synonyms, file=pkl_f)
df.to_csv(f'{WORDS_FOLDER}/openwordnet/words_{args.pos}.csv',sep='\t',index=True)
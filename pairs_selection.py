import os
import pickle
import toolsIO as io
from params import WORDS_FOLDER

import argparse
parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist


model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
contexts = io.getContexts()
dist, _ = io.getDist(distance)
semchanges = io.getSemChange(pos,model_name)

with open(f'{WORDS_FOLDER}/fernald_synonyms.pickle','rb') as f:
    syns_dict = pickle.load(f)[pos]

if not os.path.exists(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/'):
    os.mkdir(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/')

# OLD: TWO CHANGED ABOVE AVG

target_changes = semchanges['1990']
threshold = target_changes.mean()
syns_pairs = []
for head, syns in syns_dict.items():
    if head in word_list:
        if target_changes[head] >= threshold:
            for syn in syns:
                if syn in word_list:
                    if target_changes[syn] >= threshold:
                        syns_pairs.append( (head,syn) )

with open(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/fernald_{pos}_{model_name}_OLD.pickle','wb') as f:
    pickle.dump(file=f,obj=syns_pairs)
print('OLD:', len(syns_pairs))

# ONE CHANGED ABOVE AVG

target_changes = semchanges['1990']
threshold = target_changes.mean() #+ target_changes.std()
syns_pairs = []
for head, syns in syns_dict.items():
    if head in word_list:
        for syn in syns:
            if syn in word_list:
                if target_changes[syn] >= threshold or target_changes[head] >= threshold:
                    syns_pairs.append( (head,syn) )

with open(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/fernald_{pos}_{model_name}_AVG.pickle','wb') as f:
    pickle.dump(file=f,obj=syns_pairs)
print('AVG:', len(syns_pairs))

# ONE CHANGED ABOVE AVG + STD

target_changes = semchanges['1990']
threshold = target_changes.mean() + target_changes.std()
syns_pairs = []
for head, syns in syns_dict.items():
    if head in word_list:
        for syn in syns:
            if syn in word_list:
                if target_changes[syn] >= threshold or target_changes[head] >= threshold:
                    syns_pairs.append( (head,syn) )

with open(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/fernald_{pos}_{model_name}_AVGSTD.pickle','wb') as f:
    pickle.dump(file=f,obj=syns_pairs)
print('AVGSTD:', len(syns_pairs))


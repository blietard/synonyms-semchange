import pandas as pd 
import numpy as np
from params import WORDS_FOLDER, DECADES, INFO_WORDS_FOLDER
import toolsIO as io 
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Compute the semantic change for every decades.')
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

targets_df = pd.read_csv(WORDS_FOLDER + 'candidates_target_list.csv',delimiter='\t')
freq_series = targets_df.groupby('PoS').get_group(pos).set_index(['word-cs','decade'])['freq']

table = np.empty(shape=(len(word_list),len(DECADES)),dtype='int32')
for i,target in tqdm(enumerate(word_list)):
    table[i] = [freq_series.loc[(target,d)] for d in DECADES]

df = pd.DataFrame(data=table,index=word_list,columns=DECADES)
df.index.name = 'words'
df.to_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_frequency.csv', sep='\t',index=True)
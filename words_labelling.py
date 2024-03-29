import pandas as pd 
import numpy as np
from params import DECADES, NB_FREQ_GROUPS, INFO_WORDS_FOLDER
import toolsIO as io 
import os
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

import argparse
parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist


model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
freqs = io.getFreq(pos,model_name)
semchanges = io.getSemChange(pos,model_name)
semchanges_OP = io.getSemChange(pos, model_name, procrustes=True)

origin_decade = DECADES[0]
end_decade = DECADES[-1]
nb_words = len(word_list)

sc = semchanges[str(end_decade)]
sc_OP = semchanges_OP[str(end_decade)]
origin_freqs = freqs[str(origin_decade)]
end_freqs = freqs[str(end_decade)]

average_sc = sc.mean()
deviation_sc = sc.std()

average_sc_OP = sc_OP.mean()
deviation_sc_OP = sc_OP.std()

origin_freq_df = origin_freqs.sort_values().to_frame()
origin_freq_df['F_rank'] = range(1,nb_words+1)
origin_freq_df['group'] = 0
threshold = nb_words // 2
for _ in range(NB_FREQ_GROUPS-1):
    origin_freq_df.loc[ origin_freq_df.F_rank >= threshold, 'group' ] += 1
    threshold =  (nb_words+threshold) / 2
origin_freq_df = origin_freq_df.loc[word_list]

end_freq_df = end_freqs.sort_values().to_frame()
end_freq_df['F_rank'] = range(1,nb_words+1)
end_freq_df['group'] = 0
threshold = nb_words // 2
for _ in range(NB_FREQ_GROUPS-1):
    end_freq_df.loc[ end_freq_df.F_rank >= threshold, 'group' ] += 1
    threshold =  (nb_words+threshold) // 2
end_freq_df = end_freq_df.loc[word_list]


columns = ['Origin_Freq','End_Freq','FGO','FGE']
df = pd.DataFrame(columns=columns, data = np.vstack([origin_freqs ,end_freqs ,origin_freq_df['group'].values, end_freq_df['group'].values]).T ,  index=word_list)
df.index.name = 'words'

sc_group = []
for word in word_list:
    if sc.loc[word] > average_sc:
        if sc.loc[word] > average_sc + deviation_sc:
            sc_group.append('avgstd')
        else:
            sc_group.append('avg')
    else:
        sc_group.append('')

df['DDG_N'] = sc_group
df['DDG_N_avg'] = sc > average_sc
df['DDG_N_avgstd'] = sc > average_sc + deviation_sc

sc_group_op = []
for word in word_list:
    if sc_OP.loc[word] > average_sc_OP:
        if sc_OP.loc[word] > average_sc_OP + deviation_sc_OP:
            sc_group_op.append('avgstd')
        else:
            sc_group_op.append('avg')
    else:
        sc_group_op.append('')
df['DDG_OP'] = sc_group_op
df['DDG_OP_avg'] = sc_OP > average_sc_OP
df['DDG_OP_avgstd'] = sc_OP > average_sc_OP + deviation_sc_OP

df['FreqEvol'] =  df['FGE'] - df['FGO']

df.to_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_wordlabels.csv', sep='\t',index=True)
print(f'[INFO:] (Neighbors) Avg SC : {average_sc}, Avg SC + Std : {average_sc + deviation_sc}')
print(f'[INFO:] (OP) Avg SC : {average_sc_OP}, Avg SC + Std : {average_sc_OP + deviation_sc_OP}')
print(f'[INFO:] CSV stored in "{INFO_WORDS_FOLDER}/{model_name}/{pos}_wordlabels.csv"')
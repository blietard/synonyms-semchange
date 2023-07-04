import numpy as np
import pandas as pd 
from scipy import stats
import argparse

import toolsIO as io 
from params import DECADES, COLORMAP, IMG_FOLDER, SND_K_RANGE

import matplotlib.pyplot as plt
from IPython.display import display, Math

from sklearn.metrics import fbeta_score, roc_auc_score, matthews_corrcoef, log_loss, accuracy_score, make_scorer

parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V','ALL'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist
model_name = repr_mode + '_' + distance

word_list = []
synpairs_df_list = []
if pos=='ALL':
    for postag in ['ADJ','N','V']:
        word_list += io.getTargets(postag,repr_mode)[0]
        synpairs_df_list.append(io.getSynpairsInfos(postag, model_name))
    synpairs_df = pd.concat(synpairs_df_list)
else:
    word_list =  io.getTargets(pos,repr_mode)[0]
    synpairs_df = io.getSynpairsInfos(pos, model_name)



ld_IC_OP_avg = (synpairs_df.DDG_OP_avg_one & ~synpairs_df.DDG_OP_avg_both)


print('\n======= PERFORMANCE EVALUATION =======\n')

freq_t = 0
synpairs_df['FreqPred'] = 'syns'
ld_freq = ( ((synpairs_df.FGEv_entry < -freq_t)&(synpairs_df.FGEv_syn >= -freq_t)&(synpairs_df.FGDE < 0)) |
        ((synpairs_df.FGEv_syn < -freq_t)&(synpairs_df.FGEv_entry >= -freq_t)&(synpairs_df.FGDE > 0)))
synpairs_df.loc[ld_freq, 'FreqPred'] = 'diff'


div_methods = ['DivG_avg',] + ['NDivG_avg_'+str(k) for k in SND_K_RANGE] + [ 'DivG_avgstd',] + ['NDivG_avgstd_'+str(k) for k in SND_K_RANGE]
methods = div_methods + ['FreqPred']


ts = np.array([((synpairs_df[method]=='syns') & (synpairs_df.syns_WordNet)).sum() for method in methods ])
td = np.array([((synpairs_df[method]=='diff') & (~synpairs_df.syns_WordNet)).sum() for method in methods ])
tsr = ts / synpairs_df.syns_WordNet.sum()
tdr = td / (~synpairs_df.syns_WordNet).sum()
spv = ts / np.array([(synpairs_df[method]=='syns').sum() for method in methods ])
dpv = td / np.array([(synpairs_df[method]=='diff').sum() for method in methods ])
acc = (ts+td)/len(synpairs_df)
f1 = 2*spv*tsr/(spv+tsr)
revf1 = 2*dpv*tdr/(dpv+tdr)
ba = (tsr+tdr)/2
average_precision = (spv+dpv)/2
scores_labels = ['Acc','$F_1$(s)','$F_1$(d)','BA','AP']
scores = np.array([acc,f1,revf1,ba,average_precision]).T
scores_df = pd.DataFrame(data=scores, columns=scores_labels,index=methods)

display(scores_df.round(2)[['$F_1$(s)','$F_1$(d)','BA']])

print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|ccc}\\toprule")
print(pos.upper()  + " & " + ' & '.join(scores_labels[1:-1]) + " \\\\" )
for row_i, row in scores_df.round(2)[['$F_1$(s)','$F_1$(d)','BA']].iterrows():
    print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")


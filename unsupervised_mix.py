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
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist
model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)

synpairs_df = io.getSynpairsInfos(pos, model_name)

print('\n======= CONFUSION MATRIX =======\n')

ld_Div = synpairs_df.DivG_avgstd == 'high'

ld_IC_OP_avg = (synpairs_df.DDG_OP_avg_one & ~synpairs_df.DDG_OP_avg_both)

freq_t=0
ld_freq = ~( ((synpairs_df.FGEv_entry < -freq_t)&(synpairs_df.FGEv_syn >= -freq_t)&(synpairs_df.FGDE < 0)) |
        ((synpairs_df.FGEv_syn < -freq_t)&(synpairs_df.FGEv_entry >= -freq_t)&(synpairs_df.FGDE > 0))
  )


preds = ~( ld_Div | ld_IC_OP_avg | ld_freq )

crosstab = pd.crosstab(preds, synpairs_df.syns_WordNet)
crosstab['Total'] = crosstab.sum(axis=1)
crosstab.loc['Total'] = crosstab.sum(axis=0)
display(crosstab)
print(f' $F_1={fbeta_score(y_pred=preds, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds, y_true=synpairs_df.syns_WordNet).round(2)}$')


print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|ccc}\\toprule")
print(pos.upper() + f' $F_1={fbeta_score(y_pred=preds, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds, y_true=synpairs_df.syns_WordNet).round(2)}$' + " & \multicolumn{2}{c}{is in WN?} & \\\\" )
print('$\\alpha=0$ & ' + " & ".join(list(crosstab.columns.astype(str))) + ' \\\\\\hline')
for row_i, row in crosstab.iterrows():
    print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")


import numpy as np
import pandas as pd 
from scipy import stats
import argparse

import toolsIO as io 
from params import DECADES, COLORMAP, IMG_FOLDER

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

crosstab = pd.crosstab(synpairs_df.Dec_avg, synpairs_df.syns_WordNet)
crosstab['Total'] = crosstab.sum(axis=1)
crosstab.loc['Total'] = crosstab.sum(axis=0)

display(crosstab)

print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|ccc}\\toprule")
print(pos.upper() + f' $F_1={fbeta_score(y_pred=synpairs_df.Dec_avg == "LPC", y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$' + " & \multicolumn{2}{c}{is in WN?} & \\\\" )
print('$\\alpha=0$ & ' + " & ".join(list(crosstab.columns.astype(str))) + ' \\\\\\hline')
for row_i, row in crosstab.iterrows():
    print(row_i + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")



print('\n======= PLOTS =======\n')

distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
div_pop = distances_end - distances_origin
close_div_pop = div_pop[distances_origin < distances_origin.mean()] # Keep only closer than average pairs
nnz_close_div_pop = close_div_pop[np.nonzero(close_div_pop)] # Remove pairs with the same word

fig, ax = plt.subplots()
synpairs_df[synpairs_df.syns_WordNet].Div.plot.hist(alpha=0.6, density=True, bins=20, label='Still synonyms')
synpairs_df[~synpairs_df.syns_WordNet].Div.plot.hist(alpha=0.6, density=True, bins=20, label='Differentiated')
ymin, ymax = plt.ylim()
ax.vlines(x=nnz_close_div_pop.mean(), ymin=ymin, ymax=ymax, label='$\\tau_\\alpha$ with $\\alpha=0$', colors='black')
ax.vlines(x=nnz_close_div_pop.mean()+nnz_close_div_pop.std(), ymin=ymin, ymax=ymax, label='$\\tau_\\alpha$ with $\\alpha=1$', linestyles='--', colors='black')
ax.set_ylabel('Density')
ax.set_xlabel('Divergence $\\Delta = SDE - SDO $')
ax.legend(loc='upper left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div.png")

fig, ax = plt.subplots()
synpairs_df[synpairs_df.syns_WordNet].SDO.plot.hist(alpha=0.6, density=True, bins=20, label='Still synonyms')
synpairs_df[~synpairs_df.syns_WordNet].SDO.plot.hist(alpha=0.6, density=True, bins=20, label='Differentiated')
ymin, ymax = plt.ylim()
ax.set_ylabel('Density')
ax.set_xlabel('SDO')
ax.legend(loc='upper left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_SDO.png")

fig, ax = plt.subplots()
synpairs_df[synpairs_df.syns_WordNet].SDE.plot.hist(alpha=0.6, density=True, bins=20, label='Still synonyms')
synpairs_df[~synpairs_df.syns_WordNet].SDE.plot.hist(alpha=0.6, density=True, bins=20, label='Differentiated')
ymin, ymax = plt.ylim()
ax.set_ylabel('Density')
ax.set_xlabel('SDE')
ax.legend(loc='upper left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_SDE.png")

print('[INFO] Figures saved in '+IMG_FOLDER)

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

preds_N_avg = ~(synpairs_df.DDG_N_avg_one & ~synpairs_df.DDG_N_avg_both)
crosstab = pd.crosstab(preds_N_avg, synpairs_df.syns_WordNet)
crosstab['Total'] = crosstab.sum(axis=1)
crosstab.loc['Total'] = crosstab.sum(axis=0)
display(crosstab)
print(f' $F_1={fbeta_score(y_pred=preds_N_avg, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds_N_avg, y_true=synpairs_df.syns_WordNet).round(2)}$')

preds_OP_avg = ~(synpairs_df.DDG_OP_avg_one & ~synpairs_df.DDG_OP_avg_both)
crosstab_OP = pd.crosstab(preds_OP_avg, synpairs_df.syns_WordNet)
crosstab_OP['Total'] = crosstab_OP.sum(axis=1)
crosstab_OP.loc['Total'] = crosstab_OP.sum(axis=0)
display(crosstab_OP)
print(f' $F_1={fbeta_score(y_pred=preds_OP_avg, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds_OP_avg, y_true=synpairs_df.syns_WordNet).round(2)}$')

preds_OP_avgstd = ~(synpairs_df.DDG_OP_avgstd_one & ~synpairs_df.DDG_OP_avgstd_both)
crosstab_OP = pd.crosstab(preds_OP_avgstd, synpairs_df.syns_WordNet)
crosstab_OP['Total'] = crosstab_OP.sum(axis=1)
crosstab_OP.loc['Total'] = crosstab_OP.sum(axis=0)
display(crosstab_OP)
print(f' $F_1={fbeta_score(y_pred=preds_OP_avgstd, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds_OP_avgstd, y_true=synpairs_df.syns_WordNet).round(2)}$')


print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|ccc}\\toprule")
print(pos.upper() + f' $F_1={fbeta_score(y_pred=preds_N_avg, y_true=synpairs_df.syns_WordNet, beta=1).round(2)}$ and $MCC={matthews_corrcoef(y_pred=preds_N_avg, y_true=synpairs_df.syns_WordNet).round(2)}$' + " & \multicolumn{2}{c}{is in WN?} & \\\\" )
print('$\\alpha=0$ & ' + " & ".join(list(crosstab.columns.astype(str))) + ' \\\\\\hline')
for row_i, row in crosstab.iterrows():
    print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")



print('\n======= PLOTS =======\n')


semchanges = io.getSemChange(pos, model_name)
semchanges_OP = io.getSemChange(pos, model_name, procrustes=True)
sc = semchanges[str(DECADES[-1])]
sc_OP = semchanges_OP[str(DECADES[-1])]


# TPR and TNR for different alpha with Neighbors
min_std_times = 3
max_std_times = 4
n_points = 100
thresholds = np.linspace(
                start = sc.min(),
                stop = sc.max(),
                num = n_points
                )




tp = np.array( [ (( ~( ( (synpairs_df.DD_N_entry > t )&(synpairs_df.DD_N_syn <= t) ) | ( (synpairs_df.DD_N_syn > t )&(synpairs_df.DD_N_entry <= t) )  )
 )&(synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tn = np.array( [ (( ( ( (synpairs_df.DD_N_entry > t )&(synpairs_df.DD_N_syn <= t) ) | ( (synpairs_df.DD_N_syn > t )&(synpairs_df.DD_N_entry <= t) )  )
 )&(~synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tpr = tp / synpairs_df.syns_WordNet.sum()*100
tnr = tn / (~synpairs_df.syns_WordNet).sum()*100

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.plot(thresholds,tpr,label='TPR', color = 'goldenrod', marker='o', markevery=8, markersize=8)
ax.plot(thresholds,tnr,label='TNR', color = 'mediumpurple', marker='^', markevery=8, markersize=8)
ax.set_ylim(0, 100)

ax.vlines(x=[sc.mean(), sc.mean()+sc.std()], ymin = 0, ymax = 100 ,label='Used thresholds$',color='black',linestyle='--',alpha=0.4)

ax.yaxis.grid()
ax.set_title(f'TPR and TNR for Individual Change-based method \nfor POS {pos}, model {model_name}.')
ax.set_xlabel('Threshold for IC')
ax.set_ylabel('%')
ax.legend(loc='center left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Indiv_Rates.png")


# TPR and TNR for different alpha with OP
min_std_times = 3
max_std_times = 4
n_points = 100
thresholds = np.linspace(
                start = sc_OP.min(),
                stop = sc_OP.max(),
                num = n_points
                )

tp = np.array( [ (( ~( ( (synpairs_df.DD_OP_entry > t )&(synpairs_df.DD_OP_syn <= t) ) | ( (synpairs_df.DD_OP_syn > t )&(synpairs_df.DD_OP_entry <= t) )  )
 )&(synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tn = np.array( [ (( ( ( (synpairs_df.DD_OP_entry > t )&(synpairs_df.DD_OP_syn <= t) ) | ( (synpairs_df.DD_OP_syn > t )&(synpairs_df.DD_OP_entry <= t) )  )
 )&(~synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tpr = tp / synpairs_df.syns_WordNet.sum()*100
tnr = tn / (~synpairs_df.syns_WordNet).sum()*100

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.plot(thresholds,tpr,label='TPR', color = 'goldenrod', marker='o', markevery=8, markersize=8)
ax.plot(thresholds,tnr,label='TNR', color = 'mediumpurple', marker='^', markevery=8, markersize=8)
ax.set_ylim(0, 100)

ax.vlines(x=[sc_OP.mean(), sc_OP.mean()+sc_OP.std()], ymin = 0, ymax = 100 ,label='Used thresholds$',color='black',linestyle='--',alpha=0.4)

ax.yaxis.grid()
ax.set_title(f'TPR and TNR for Individual Change-based method \nfor POS {pos}, model {model_name}.')
ax.set_xlabel('Threshold for IC')
ax.set_ylabel('%')
ax.legend(loc='center left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Indiv_OP_Rates.png")

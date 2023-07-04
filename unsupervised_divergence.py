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

crosstab = pd.crosstab(synpairs_df.DivG_avg, synpairs_df.syns_WordNet)
crosstab['Total'] = crosstab.sum(axis=1)
crosstab.loc['Total'] = crosstab.sum(axis=0)

display(crosstab)

print('\n======= PERFORMANCE EVALUATION =======\n')


div_methods = ['DivG_avg',] + ['NDivG_avg_'+str(k) for k in SND_K_RANGE] + [ 'DivG_avgstd',] + ['NDivG_avgstd_'+str(k) for k in SND_K_RANGE]
ts = np.array([((synpairs_df[method]=='syns') & (synpairs_df.syns_WordNet)).sum() for method in div_methods ])
td = np.array([((synpairs_df[method]=='diff') & (~synpairs_df.syns_WordNet)).sum() for method in div_methods ])
tsr = ts / synpairs_df.syns_WordNet.sum()
tdr = td / (~synpairs_df.syns_WordNet).sum()
spv = ts / np.array([(synpairs_df[method]=='syns').sum() for method in div_methods ])
dpv = td / np.array([(synpairs_df[method]=='diff').sum() for method in div_methods ])

acc = (ts+td)/len(synpairs_df)
f1 = 2*spv*tsr/(spv+tsr)
revf1 = 2*dpv*tdr/(dpv+tdr)
ba = (tsr+tdr)/2
average_precision = (spv+dpv)/2
scores_labels = ['Acc','F1','revF1','BA','AP']
scores = np.array([acc,f1,revf1,ba,average_precision]).T
scores_df = pd.DataFrame(data=scores, columns=scores_labels,index=div_methods)

display(scores_df.round(2))

print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|ccc}\\toprule")
print(pos.upper() + f' $F_1={round(fbeta_score(y_pred=synpairs_df.DivG_avg == "low", y_true=synpairs_df.syns_WordNet, beta=1),2)}$ and $MCC={round(matthews_corrcoef(y_pred=synpairs_df.DivG_avg == "low", y_true=synpairs_df.syns_WordNet),2)}$' + " & \multicolumn{2}{c}{is in WN?} & \\\\" )
print('$\\alpha=0$ & ' + " & ".join(list(crosstab.columns.astype(str))) + ' \\\\\\hline')
for row_i, row in crosstab.iterrows():
    print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")



print('\n======= PLOTS =======\n')

# Divergence and why it's not working
distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
div_pop = distances_end - distances_origin
close_div_pop = div_pop[distances_origin < distances_origin.mean()] # Keep only closer than average pairs
# nnz_close_div_pop = close_div_pop[np.nonzero(close_div_pop)] # Remove pairs with the same word

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(10,8))
synpairs_df[synpairs_df.syns_WordNet].Div.plot.hist(alpha=0.6, density=True, bins=20, label='Still synonyms')
synpairs_df[~synpairs_df.syns_WordNet].Div.plot.hist(alpha=0.6, density=True, bins=20, label='Differentiated')
ymin, ymax = plt.ylim()
ax.vlines(x=div_pop.mean(), ymin=ymin, ymax=ymax, label='Computed $\\tau$', colors='black')
# ax.vlines(x=nnz_close_div_pop.mean()+nnz_close_div_pop.std(), ymin=ymin, ymax=ymax, label='$\\tau_\\alpha$ with $\\alpha=1$', linestyles='--', colors='black')
ax.set_ylabel('Density')
ax.set_xlabel('$\\Delta = SD^{(T2)} - SD^{(T1)} $')
ax.set_xlim(-0.6,0.6)
ax.legend(loc='upper left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div.png")

for k in SND_K_RANGE:
    fig, ax = plt.subplots()
    synpairs_df[synpairs_df.syns_WordNet]['NDiv_'+str(k)].plot.hist(alpha=0.6, density=True, bins=20, label='Still synonyms')
    synpairs_df[~synpairs_df.syns_WordNet]['NDiv_'+str(k)].plot.hist(alpha=0.6, density=True, bins=20, label='Differentiated')
    ymin, ymax = plt.ylim()
    ax.set_ylabel('Density')
    ax.set_xlabel('neighborhood Divergence $\\Delta = SNDE - SNDO $')
    ax.legend(loc='upper left')
    fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div_Neigh_{k}.png")


# SDO and SDE to have a look
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



# TPR and TNR for different alpha with 
min_std_times = 3
max_std_times = 4
n_points = 100
thresholds = np.linspace(
                start = nnz_close_div_pop.mean() - min_std_times*nnz_close_div_pop.std(),
                stop = nnz_close_div_pop.mean() + max_std_times*nnz_close_div_pop.std(),
                num = n_points
                )

tp = np.array( [ ((synpairs_df.Div <= t)&(synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tn = np.array( [ ((synpairs_df.Div > t)&(~synpairs_df.syns_WordNet)).sum() for t in thresholds] )
tpr = tp / synpairs_df.syns_WordNet.sum()*100
tnr = tn / (~synpairs_df.syns_WordNet).sum()*100

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.plot(thresholds,tpr,label='True Syns Rate', color = 'goldenrod', marker='o', markevery=8, markersize=8)
ax.plot(thresholds,tnr,label='True Diff Rate', color = 'mediumpurple', marker='^', markevery=8, markersize=8)
ax.set_ylim(0, 100)

ax.vlines(x=[nnz_close_div_pop.mean(), nnz_close_div_pop.mean()+nnz_close_div_pop.std()], ymin = 0, ymax = 100 ,label='Used $\\alpha$',color='black',linestyle='--',alpha=0.4)

ax.set_xticks( ticks=[nnz_close_div_pop.mean() + i*nnz_close_div_pop.std() for i in range(-min_std_times, max_std_times+1)], labels=list(range(-min_std_times, max_std_times+1)) )

ax.yaxis.grid()
#ax.set_title(f'TPR and TNR for Divergence-based method \nfor POS {pos}, model {model_name}.')
ax.set_xlabel('Threshold parameter $\\alpha$')
ax.set_ylabel('%')
ax.legend(loc='center left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div_Rates.png")

# TPR and TNR for different alpha
n_points = 100
for k in SND_K_RANGE:
    thresholds = np.linspace(
                    start = synpairs_df['NDiv_'+str(k)].min(),
                    stop = synpairs_df['NDiv_'+str(k)].max(),
                    num = n_points
                    )

    tp = np.array( [ ((synpairs_df['NDiv_'+str(k)] <= t)&(synpairs_df.syns_WordNet)).sum() for t in thresholds] )
    tn = np.array( [ ((synpairs_df['NDiv_'+str(k)] > t)&(~synpairs_df.syns_WordNet)).sum() for t in thresholds] )
    tpr = tp / synpairs_df.syns_WordNet.sum()*100
    tnr = tn / (~synpairs_df.syns_WordNet).sum()*100

    fig = plt.figure(figsize=(12,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)

    ax.plot(thresholds,tpr,label='True Syns Rate', color = 'goldenrod', marker='o', markevery=8, markersize=8)
    ax.plot(thresholds,tnr,label='True Diff Rate', color = 'mediumpurple', marker='^', markevery=8, markersize=8)
    ax.set_ylim(0, 100)

    ax.yaxis.grid()
    #ax.set_title(f'TPR and TNR for Neighborhood Divergence-based method \nfor POS {pos}, model {model_name} and k={k}.')
    ax.set_xlabel('Neighborhood Divergence')
    ax.set_ylabel('%')
    ax.legend(loc='center left')
    fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div_Rates_Neigh_{k}.png")


# Rates of True Diff/Syns with respect to k for tau=avg
tSr = [((synpairs_df['NDivG_avg_'+str(k)] == 'syns') & (synpairs_df.syns_WordNet)).sum()*100/synpairs_df.syns_WordNet.sum() for k in SND_K_RANGE]
tDr = [((synpairs_df['NDivG_avg_'+str(k)] == 'diff') & (~synpairs_df.syns_WordNet)).sum()*100/(~synpairs_df.syns_WordNet).sum() for k in SND_K_RANGE]
fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)
ax.plot(range(len(SND_K_RANGE)),tDr,label='True Diff Rate', color = 'goldenrod', marker='o', markevery=1, markersize=8)
ax.plot(range(len(SND_K_RANGE)),tSr,label='True Syns Rate', color = 'mediumpurple', marker='^', markevery=1, markersize=8)
ax.set_ylim(0, 100)
ax.set_xticks( ticks=range(len(SND_K_RANGE)), labels=SND_K_RANGE )
ax.yaxis.grid()
ax.set_title(f'Influence of $k$ over True Rates of Diff and Syns \nfor Divergence-based method with Neighbors-SD \nfor POS {pos}, model {model_name}.')
ax.set_xlabel('$k$')
ax.set_ylabel('%')
ax.legend(loc='center left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div_Rates_avg_neigh.png")

# Rates of True Diff/Syns with respect to k for tau=avgstd
tSr = [((synpairs_df['NDivG_avgstd_'+str(k)] == 'syns') & (synpairs_df.syns_WordNet)).sum()*100/synpairs_df.syns_WordNet.sum() for k in SND_K_RANGE]
tDr = [((synpairs_df['NDivG_avgstd_'+str(k)] == 'diff') & (~synpairs_df.syns_WordNet)).sum()*100/(~synpairs_df.syns_WordNet).sum() for k in SND_K_RANGE]
fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)
ax.plot(range(len(SND_K_RANGE)),tDr,label='True Diff Rate', color = 'goldenrod', marker='o', markevery=1, markersize=8)
ax.plot(range(len(SND_K_RANGE)),tSr,label='True Syns Rate', color = 'mediumpurple', marker='^', markevery=1, markersize=8)
ax.set_ylim(0, 100)
ax.set_xticks( ticks=range(len(SND_K_RANGE)), labels=SND_K_RANGE )
ax.yaxis.grid()
ax.set_title(f'Influence of $k$ over True Rates of Diff and Syns \nfor Divergence-based method with Neighbors-SD \nfor POS {pos}, model {model_name}.')
ax.set_xlabel('$k$')
ax.set_ylabel('%')
ax.legend(loc='center left')
fig.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_unsup_Div_Rates_avgstd_neigh.png")


print('[INFO] Figures saved in '+IMG_FOLDER)


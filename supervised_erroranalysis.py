import numpy as np
import pandas as pd 
from scipy import stats
import argparse
from tqdm import tqdm

import toolsIO as io 
from params import DECADES, COLORMAP, SND_K_RANGE, SUPERVISED_TRIALS, IMG_FOLDER

import matplotlib.pyplot as plt
from IPython.display import display, Math

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

def truePos(y_true, y_preds):
    return np.sum( (y_preds == True) & (y_true == True) )

def trueNeg(y_true, y_preds):
    return np.sum( (y_preds == False) & (y_true == False) )
def get_scores(y_preds,y_true):

    tp = truePos(y_true, y_preds)
    tn = trueNeg(y_true, y_preds)
    p = y_true.sum()
    n = (~y_true).sum()
    pp = y_preds.sum()
    pn = (~y_preds).sum()
    
    tpr = tp / p
    tnr = tn / n
    ppv = tp / max(pp,1)
    npv = tn / max(pn,1)
    f1 = max(0,2*ppv*tpr/(ppv+tpr))
    revf1 = max(0,2*npv*tnr/(npv+tnr))
    ba = (tpr+tnr)/2
    average_precision = (ppv+npv)/2
    scores_names = ['PP','PN','F1 (syns)','F1 (diff)','BA','AP']
    scores_values = np.array([pp,pn,f1,revf1,ba,average_precision])
    return (scores_values,scores_names)

class SynonymyClassif:

    def __init__(self, clf_model, features, kwargs_dict={}, preprocessor=None):
        self.kwargs = kwargs_dict
        self.model_constructor = clf_model
        self.model = clf_model(**kwargs_dict)
        self.features = features
        self.preprocessor = preprocessor

    def fit(self,X,y):
        X_train = X[self.features]
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
        self.model.fit(X_train, y)

    def predict(self,X):
        X = X[self.features]
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)

    def score(self,X,y):
        return get_scores(y_preds=self.predict(X),y_true=y)





# Datasets creation
X = synpairs_df.copy()
Y = synpairs_df.syns_WordNet

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=None, shuffle=True)

#===== Logistic Regression =====
name = 'LR multivariate'
clf = SynonymyClassif(
                clf_model = LogisticRegression,
                features = ['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],
                kwargs_dict = {'class_weight':'balanced','max_iter':500, 'penalty':'none'},
                preprocessor=StandardScaler()
                )
clf.fit(X_train,y_train)
this_name = name
score_values, score_names = clf.score(X_test,y_test)
print('\n'.join([f'{name} : {round(score,2)}' for score, name in zip(score_values,score_names) ]))
y_preds = pd.Series(clf.predict(X_test), name='y_preds', index=X_test.index)

tp = y_preds & y_test
fp = y_preds & ~y_test
tn = ~y_preds & ~y_test
fn = ~y_preds & y_test

X_test['BestPredictor'] = None
X_test.loc[tp, 'BestPredictor'] = 'TP'
X_test.loc[fp, 'BestPredictor'] = 'FP'
X_test.loc[tn, 'BestPredictor'] = 'TN'
X_test.loc[fn, 'BestPredictor'] = 'FN'



for feature in ['SDO','pressure_entry','WNsenses_entry','WNsenses_syn','DD_OP_entry','DD_OP_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn']:
    print('\n============ ' + feature + ' ============')
    df = X_test.groupby('BestPredictor')[feature].agg(['mean','std','min','max','size']).round(2)
    display(df)

    print('\n======= AS LATEX =======\n')

    print("\\begin{tabular}{c|cccccc}\\toprule")
    print(f"{pos.upper()} {feature} & " + ' & '.join(df.columns) + " \\\\" )
    for row_i, row in df.iterrows():
        print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
    print("\\bottomrule\n\\end{tabular}")
    print('\n========================')


X_test.loc[X_test.WN_distance==np.inf, 'WN_distance']= np.NaN

print('-- Significant differences --')
for feature in ['SDO','pressure_entry','WNsenses_entry','WN_hypernymy','WNsenses_syn','WN_distance','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn']:
    print('\n============ ' + feature + ' ============\n')
    
    tp = X_test.groupby('BestPredictor')[feature].get_group('TP').dropna()
    tn = X_test.groupby('BestPredictor')[feature].get_group('TN').dropna()
    fp = X_test.groupby('BestPredictor')[feature].get_group('FP').dropna()
    fn = X_test.groupby('BestPredictor')[feature].get_group('FN').dropna()
    print('Means (PP/PN) : ', round( pd.concat([tp,fp]).mean() ,2), round(pd.concat([tn,fn]).mean(),2))
    print('tTest: ', round(stats.ttest_ind(pd.concat([tp,fp]),pd.concat([tn,fn]))[1],3))
    print('MW-U: ', round(stats.mannwhitneyu(pd.concat([tp,fp]),pd.concat([tn,fn]))[1],3))
    print('Means (TP/FN) : ', round(tp.mean(),2), round(fn.mean(),2))
    print('tTest: ', round(stats.ttest_ind(tp,fn)[1],3))
    print('MW-U: ', round(stats.mannwhitneyu(tp,fn)[1],3))
    print('Means (TN/FP) : ', round(tn.mean(),2), round(fp.mean(),2))
    print('tTest: ', round(stats.ttest_ind(tn,fp)[1],3))
    print('MW-U: ', round(stats.mannwhitneyu(tn,fp)[1],3))
    





max_displayed_WNdist = 3

all_counts = []

for _ in tqdm(range(SUPERVISED_TRIALS)):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=None, shuffle=True)

    #===== Logistic Regression =====
    clf = SynonymyClassif(
                    clf_model = LogisticRegression,
                    features = ['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],
                    kwargs_dict = {'class_weight':'balanced','max_iter':500, 'penalty':'none'},
                    preprocessor=StandardScaler()
                    )
    clf.fit(X_train,y_train)
    y_preds = pd.Series(clf.predict(X_test), name='y_preds', index=X_test.index)
    counts = []
    for i in range(max_displayed_WNdist+1):
        mask = X_test.WN_distance == i
        counts.append((y_preds[mask].value_counts().sort_index(ascending=False)/sum(mask)*100).values)
    mask = X_test.WN_distance > max_displayed_WNdist
    counts.append((y_preds[mask].value_counts().sort_index(ascending=False)/sum(mask)*100).values)
    counts_array = np.array(counts).T
    all_counts.append(counts_array)

all_counts = np.array(all_counts).mean(axis=0)


plt.rcParams.update({'font.size': 22})

colors = ['tab:blue','tab:orange']
labels = ['Syns','Diff']
x = np.arange(max_displayed_WNdist+2)  # the label locations
width = 0.7  # the width of the bars

fig, ax = plt.subplots(figsize=(10,10))
fig.set_facecolor('white')
bottom = 0
for i in range(all_counts.shape[0]):
    rect = ax.bar(x, all_counts[i].round(1) , width, label=labels[i], color=colors[i], bottom=bottom)
    ax.bar_label(rect, padding=3, label_type='center')
    bottom += all_counts[i]

ax.set_xlabel('Distance in WordNet\'s graph')
ax.set_ylabel('Rate of predictions (%)')
#ax.set_title('True Positive / True Negative rates with $\\alpha=0$ (avg) or $\\alpha=1$ (avgstd)')
ax.set_xticks(x, list(range(max_displayed_WNdist+1))+[f'>{max_displayed_WNdist}'])
ax.legend(loc='upper right')
plt.savefig(f"{IMG_FOLDER}/{model_name}_{pos}_BestModel_WNdistPreds.png")



# print('\n========== AS LATEX ============\n')

# for synsSet in scores_dataFrames_per_synset.keys():
#     df = summary_df.loc[synsSet]
#     print(df.round(2).style.to_latex())
    # print( f'{synsSet} {pos} & ' + " & ".join(list(df.columns)) + ' \\\\')
    # for row_i, row in df.iterrows():
    #     if row_i in ['TP','TN','FN','FP']:
    #         print(row_i + ' & ' + " & ".join(list(row.values.astype('int').astype('str'))) + ' \\\\')
    #     else:
    #         print(row_i + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
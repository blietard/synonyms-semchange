import numpy as np
import pandas as pd 
from scipy import stats
import argparse

import toolsIO as io 
from params import DECADES, COLORMAP

import matplotlib.pyplot as plt
from IPython.display import display, Math

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, roc_auc_score, matthews_corrcoef, log_loss, accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
parser.add_argument('degree', metavar='Degree', type=int, nargs='?',
                    help='Degree')
args = parser.parse_args()
pos, repr_mode, distance, degree = args.pos, args.repr, args.dist, args.degree
model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)

synpairs_df = io.getSynpairsInfos(pos, model_name)


def truePos(y_true, y_preds):
    return np.sum( (y_preds == True) & (y_true == True) )

def trueNeg(y_true, y_preds):
    return np.sum( (y_preds == False) & (y_true == False) )

def falsePos(y_true, y_preds):
    return np.sum( (y_preds == True) & (y_true == False) )

def falseNeg(y_true, y_preds):
    return np.sum( (y_preds == False) & (y_true == True) )


def print_scores(name, scores_arr, scores_names):
    for score, scorename in zip(scores_arr, scores_names):
        print(f'{name} {scorename} : {np.round(score,3)}')

def get_scores(y_preds,y_true):
    scores_values = [truePos(y_true, y_preds), trueNeg(y_true, y_preds), falsePos(y_true, y_preds), falseNeg(y_true, y_preds) , accuracy_score(y_true, y_preds), fbeta_score(y_true, y_preds, beta=1), fbeta_score(y_true, y_preds, beta=2), roc_auc_score(y_true, y_preds), matthews_corrcoef(y_true, y_preds)]
    scores_names = ['TP','TN','FP','FN','Test Accuracy', 'F1-score','F2-score','ROC-AUC','MCC']
    return (scores_values,scores_names)

def loglikelihood(y_train,preds):
    return -log_loss(y_train, preds)*len(preds)

def aic(preds,y_train,n_features,n_train_samples):
    return -2/n_train_samples*loglikelihood(y_train,preds) + 2*n_features/n_train_samples

def bic(preds,y_train,n_features,n_train_samples):
    return -2*loglikelihood(y_train,preds)+ np.log(n_train_samples)*n_features

class DivergenceBasedClassifier:
    def __init__(self, threshold='avg'):
        self.threshold = threshold
    
    def fit(self, X,y):
        pass

    def predict(self,X):
        return ( X['Dec_'+self.threshold] == 'LPC' )

class ConstantOutputClassifier:
    def __init__(self,constant, out_type='bool'):
        self.constant = constant
        self.out_type = out_type

    def fit(self,X,y):
        pass

    def predict(self,X):
        return (np.ones(X.shape[0])*self.constant).astype(self.out_type)

class SynonymyClassif:

    def __init__(self, clf_model, features, kwargs_dict={},CVgridSearch = False, CVgridSearch_score = None, standardscale=False, polydegree=0):
        self.kwargs = kwargs_dict
        self.model_constructor = clf_model
        self.CVgrid = CVgridSearch
        self.CVgrid_score = CVgridSearch_score
        self.polydegree = polydegree
        if CVgridSearch :
            self.model = GridSearchCV(self.model_constructor(), kwargs_dict , scoring=CVgridSearch_score)
        else:
            self.model = clf_model(**kwargs_dict)
        self.features = features
        self.train_accuracy = 0
        self.standardscale = standardscale
        if self.polydegree > 1:
            self.polyfeat_Transformer = PolynomialFeatures(degree=self.polydegree)

    def fit(self,X,y):
        X_train = X[self.features]
        if self.standardscale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        if self.polydegree > 1:
            X_train = self.polyfeat_Transformer.fit_transform(X_train)
        self.model.fit(X_train, y)
        self.train_accuracy = accuracy_score(y_pred= self.predict(X),y_true=y)

    def predict(self,X):
        X = X[self.features]
        if self.standardscale:
            X = self.scaler.transform(X)
        if self.polydegree > 1:
            X = self.polyfeat_Transformer.transform(X)
        return self.model.predict(X)

    def score(self,X,y):
        return get_scores(y_preds=self.predict(X),y_true=y)

    def crosstab(self,X,y):
        return pd.crosstab(y, pd.Series(self.predict(X), name='Predictions', index = y.index ))

    def reinit(self, new_features = None):
        if new_features:
            self.__init__(self.model_constructor, new_features, self.kwargs, self.CVgrid, self.CVgrid_score, self.standardscale, self.polydegree)
        else:
            self.__init__(self.model_constructor, self.features, self.kwargs, self.CVgrid, self.CVgrid_score, self.standardscale, self.polydegree)
    
    def logreg_feature_selection(self, X, y, criterion_func):
        n_samples = y.shape[0]
        n_features = len(self.features)
        base_crit_value = criterion_func(self.predict(X),y,n_features,n_samples)
        selected_features = self.features.copy()
        best_crit_value = base_crit_value
        step_number = 0
        while True:
            crit_values = np.empty(len(selected_features))
            for i in range(len(selected_features)):
                reduced_features = selected_features[:i] + selected_features[i+1:] #exclude feature i
                self.reinit(reduced_features)
                self.fit(X,y)
                n_features = len(reduced_features)
                crit_values[i] = criterion_func(self.predict(X),y,n_features,n_samples)
            lowest_crit_value = np.min(crit_values)
            if lowest_crit_value > best_crit_value :
                break
            feature_to_remove_ind = np.argmin(crit_values)
            removed_feat = selected_features.pop(feature_to_remove_ind)
            best_crit_value = lowest_crit_value
            step_number += 1
        self.reinit(selected_features)
        self.fit(X,y)
        return {'init_crit_value' : base_crit_value, 'final_crit_value' : best_crit_value, 'nb_step' : step_number}
      


features = ['SDO','SDE','normLev','DiachrD_entry','DiachrD_syn', 'DDGpair_atleast1', 'DDGpair_both','FGO_entry','FGE_entry','FGO_syn','FGE_syn','absFEv_entry','absFEv_syn']
synsSets_to_use = ['WordNet']
minimum_TrueLabel_frequency = 0 # 0 for no sampling

resultsSummary_per_synSets = dict()
lr_features_coeff_per_synSets = dict()

for synsSet in synsSets_to_use:
    target_var = synpairs_df['syns_'+synsSet]
    target_var_counts = target_var.value_counts()
    
    # Class balance
    rng = np.random.default_rng(42)
    if target_var.mean() <= minimum_TrueLabel_frequency : # 1/3 = there can be AT MOST twice more False than True.
        p = (target_var_counts[True]*2)/target_var_counts[False]
    else:
        p = 1 # Take everything
    sample = pd.Series( rng.binomial(n=1,p=p,size=len(synpairs_df)) , index=synpairs_df.index)
    
    # Datasets creation
    X = synpairs_df.loc[ (target_var | sample) , :].copy()
    Y = target_var[ (target_var | sample) ]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
    scores_per_model = dict()

    #===== Logistic Regression =====
    name = 'L1 LR'
    print(name)
    clf = SynonymyClassif(
                    clf_model = LogisticRegression,
                    features = features,
                    kwargs_dict = {'class_weight':['balanced'],'max_iter':[1000], 'penalty':['l1'], 'solver':['liblinear'],'C':(np.arange(1,41)/20)},
                    CVgridSearch = True,
                    CVgridSearch_score = make_scorer(matthews_corrcoef),
                    standardscale=True,
                    polydegree=degree
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], scores_names = clf.score(X_test,y_test)
    print(f"Best C for regul LogReg on {synsSet} : {clf.model.best_params_['C']}")
    # print('-------- Coeff L1 LR -----------')
    # print(pd.Series(index=features,data=clf.model.best_estimator_.coef_[0].round(3)).sort_values())

    #===== Decision Tree =====
    name = 'Tree'
    print(name)
    clf = SynonymyClassif(
                    clf_model = DecisionTreeClassifier,
                    features = features,
                    kwargs_dict = {'class_weight':'balanced', 'max_depth':10 , 'criterion':'gini', 'min_samples_split':10},
                    polydegree=degree
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], scores_names = clf.score(X_test,y_test)
    
    #===== Random Forest =====
    name = 'RF'
    print(name)
    n_trials = 20
    test_scores_list = []
    for i in range(n_trials):
        clf = SynonymyClassif(
                    clf_model = RandomForestClassifier,
                    features = features,
                    kwargs_dict = {'class_weight':'balanced', 'max_depth':10 , 'criterion':'gini', 'min_samples_split':10, 'n_estimators':10},
                    polydegree=degree
                    )
        clf.fit(X_train,y_train)
        test_scores, _ = clf.score(X_test,y_test)
        test_scores_list.append(test_scores)
    scores_per_model[name] = np.mean(test_scores_list,axis=0)
    
    #===== Divergence-based (threshold = avg) ======
    name = 'Div avg'
    print(name)
    clf = SynonymyClassif(
                    clf_model = DivergenceBasedClassifier,
                    features = features + ['Dec_avg', 'Dec_avgstd'],
                    kwargs_dict = {'threshold':'avg'}
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], _ = clf.score(X_test,y_test)

    #===== Divergence-based (threshold = avgstd) ======
    name = 'Div avgstd'
    print(name)
    clf = SynonymyClassif(
                    clf_model = DivergenceBasedClassifier,
                    features = features + ['Dec_avg', 'Dec_avgstd'],
                    kwargs_dict = {'threshold':'avgstd'}
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], _ = clf.score(X_test,y_test)

    #===== Always True ======
    name = 'AllTrue'
    print(name)
    clf = SynonymyClassif(
                    clf_model = ConstantOutputClassifier,
                    features = features,
                    kwargs_dict = {'constant':1,'out_type':'bool'}
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], _ = clf.score(X_test,y_test)

    #===== Always False ======
    name = 'AllFalse'
    print(name)
    clf = SynonymyClassif(
                    clf_model = ConstantOutputClassifier,
                    features = features,
                    kwargs_dict = {'constant':0,'out_type':'bool'}
                    )
    clf.fit(X_train,y_train)
    scores_per_model[name], _ = clf.score(X_test,y_test)

    #====== Summary ======
    resultsSummary_per_synSets[synsSet] = pd.DataFrame(data=scores_per_model,index=scores_names)


for dataname, df in resultsSummary_per_synSets.items():
    df['synsSet'] = dataname
    df['scores_names'] = scores_names
summary_df = pd.concat(resultsSummary_per_synSets.values())
summary_df = summary_df.set_index(['synsSet','scores_names']).round(2)


for synsSet in resultsSummary_per_synSets.keys():
    print('=========== ' + synsSet + ' ===========')
    print('-------- Performances -----------')
    display(summary_df.loc[synsSet])

print('\n========== AS LATEX ============\n')

for synsSet in resultsSummary_per_synSets.keys():
    df = summary_df.loc[synsSet]
    print(df.round(2).style.to_latex())
    # print( f'{synsSet} {pos} & ' + " & ".join(list(df.columns)) + ' \\\\')
    # for row_i, row in df.iterrows():
    #     if row_i in ['TP','TN','FN','FP']:
    #         print(row_i + ' & ' + " & ".join(list(row.values.astype('int').astype('str'))) + ' \\\\')
    #     else:
    #         print(row_i + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
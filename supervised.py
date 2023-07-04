import numpy as np
import pandas as pd 
from scipy import stats
import argparse
from collections import defaultdict
from tqdm import tqdm

import toolsIO as io 
from params import DECADES, COLORMAP, SND_K_RANGE, SUPERVISED_TRIALS

import matplotlib.pyplot as plt
from IPython.display import display, Math

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, roc_auc_score, matthews_corrcoef, log_loss, accuracy_score, make_scorer, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

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

def falsePos(y_true, y_preds):
    return np.sum( (y_preds == True) & (y_true == False) )

def falseNeg(y_true, y_preds):
    return np.sum( (y_preds == False) & (y_true == True) )


def print_scores(name, scores_arr, scores_names):
    for score, scorename in zip(scores_arr, scores_names):
        print(f'{name} {scorename} : {np.round(score,3)}')

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
    if ppv:
        f1 = 2*ppv*tpr/(ppv+tpr)
    else:
        f1 = 0
    if npv:
        revf1 = 2*npv*tnr/(npv+tnr)
    else:
        revf1 = 0
    ba = (tpr+tnr)/2
    average_precision = (ppv+npv)/2
    scores_names = ['PD','F1 (syns)','F1 (diff)','BA','AP']
    scores_values = np.array([pn/len(y_true),f1,revf1,ba,average_precision])
    return (scores_values,scores_names)

def loglikelihood(y_train,preds):
    return -log_loss(y_train, preds)*len(preds)

def aic(preds,y_train,n_features,n_train_samples):
    return -2/n_train_samples*loglikelihood(y_train,preds) + 2*n_features/n_train_samples

def bic(preds,y_train,n_features,n_train_samples):
    return -2*loglikelihood(y_train,preds)+ np.log(n_train_samples)*n_features

class PreComputedClassifier:
    def __init__(self, SD_feature):
        self.SD_feature_name = SD_feature
    def fit(self, X,y):
        pass
    def predict(self,X):
        return ( X[self.SD_feature_name] == 'syns' )

class DivergenceBasedClassifier:
    def __init__(self, SD_feature, threshold=None, fit_scorer = fbeta_score):
        self.SD_feature_name = SD_feature
        self.tau=threshold
        self.fit_scorer = fit_scorer
    
    def fit(self, X,y):
        if self.tau:
            pass
        else:
            tau_range = np.linspace(X[self.SD_feature_name].min(), X[self.SD_feature_name].max(), 1000)
            best_tau = None
            score_max = -np.inf
            for tau in tau_range:
                preds = ( X[self.SD_feature_name] < tau )
                score = self.fit_scorer(y_true=y,y_pred=preds, beta=1, average='macro')
                if score > score_max:
                    score_max = score
                    best_tau = tau
            self.tau = best_tau

    def predict(self,X):
        return ( X[self.SD_feature_name] < self.tau )

    def get_params(self, deep=None):
        return np.array([self.tau])

class FrequencyBasedClassifier:
    def __init__(self, threshold=0):
        self.t = threshold

    def fit(self, X, y):
        pass
    
    def predict(self,X):
        return ~( ((X.FGEv_entry < -self.t)&(X.FGEv_syn >= -self.t)&(X.FGDE < 0)) |
                ((X.FGEv_syn < -self.t)&(X.FGEv_entry >= -self.t)&(X.FGDE > 0)))

class DivAndFreqCombinedClassifier:
    def __init__(self, SD_feature, div_threshold, freq_threshold, fit_scorer = fbeta_score):
        self.SD_feature_name = SD_feature
        self.tau = div_threshold
        self.t = freq_threshold
        self.fit_scorer = fit_scorer

    def fit(self, X, y):
        if self.tau:
            pass
        else:
            tau_range = np.linspace(X[self.SD_feature_name].min(), X[self.SD_feature_name].max(), 1000)
            best_tau = None
            score_max = -np.inf
            for tau in tau_range:
                div_criterion = X[self.SD_feature_name] < tau
                freq_criterion = ~( ((X.FGEv_entry < -self.t)&(X.FGEv_syn >= -self.t)&(X.FGDE < 0)) |
                    ((X.FGEv_syn < -self.t)&(X.FGEv_entry >= -self.t)&(X.FGDE > 0)))
                preds = ( X[self.SD_feature_name] < tau )
                score = self.fit_scorer(y_true=y,y_pred=preds, beta=1, average='macro')
                if score > score_max:
                    score_max = score
                    best_tau = tau
            self.tau = best_tau

    
    def predict(self,X):
        div_criterion = X[self.SD_feature_name] < self.tau
        freq_criterion = ~( ((X.FGEv_entry < -self.t)&(X.FGEv_syn >= -self.t)&(X.FGDE < 0)) |
                ((X.FGEv_syn < -self.t)&(X.FGEv_entry >= -self.t)&(X.FGDE > 0)))
        return div_criterion & freq_criterion

class ConstantOutputClassifier:
    def __init__(self,constant, out_type='bool'):
        self.constant = constant
        self.out_type = out_type

    def fit(self,X,y):
        pass

    def predict(self,X):
        return (np.ones(X.shape[0])*self.constant).astype(self.out_type)

class SynonymyClassif:

    def __init__(self, clf_model, features, kwargs_dict={},CVgridSearch = False, CVgridSearch_score = None, preprocessor=None):
        self.kwargs = kwargs_dict
        self.model_constructor = clf_model
        self.CVgrid = CVgridSearch
        self.CVgrid_score = CVgridSearch_score
        if CVgridSearch :
            self.model = GridSearchCV(self.model_constructor(), kwargs_dict , scoring=CVgridSearch_score)
        else:
            self.model = clf_model(**kwargs_dict)
        self.features = features
        self.train_accuracy = 0
        self.preprocessor = preprocessor

    def fit(self,X,y):
        X_train = X[self.features]
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
        self.model.fit(X_train, y)
        self.train_accuracy = accuracy_score(y_pred= self.predict(X),y_true=y)

    def predict(self,X):
        X = X[self.features]
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)

    def score(self,X,y):
        return get_scores(y_preds=self.predict(X),y_true=y)

    def crosstab(self,X,y):
        return pd.crosstab(y, pd.Series(self.predict(X), name='Predictions', index = y.index ))

    def reinit(self, new_features = None):
        if new_features:
            self.__init__(self.model_constructor, new_features, self.kwargs, self.CVgrid, self.CVgrid_score, self.preprocessor)
        else:
            self.__init__(self.model_constructor, self.features, self.kwargs, self.CVgrid, self.CVgrid_score, self.preprocessor)
    
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
    
    def prune_decision_tree(self,X,y,evaluation_metric):
        pass    



features = synpairs_df.columns
synsSets_to_use = ['WordNet']
minimum_TrueLabel_frequency = 0 # 0 for no sampling

scores_dataFrames_per_synset = defaultdict(list)
lr_features_coeff_per_synSets = dict()

for synsSet in synsSets_to_use:

    # target_var = synpairs_df['syns_'+synsSet]
    target_var = synpairs_df['syns_'+synsSet] | (synpairs_df['WN_distance']==1)
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

    scores_per_model = dict()

    for _ in tqdm(range(SUPERVISED_TRIALS)):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=None, shuffle=True)
        
        #===== Logistic Regression SD(cd) =====
        name = 'LR '
        # features_sets = [(['SDO','SDE'],'SD (cd)'),
        #                 (['SDO','SDE','DD_N_entry','DD_N_syn'], 'SD(cd)+IC(n)'),
        #                 (['SDO','SDE','DD_OP_entry','DD_OP_syn'],'SD(cd)+IC(OP)'),
        #                 (['SDO','SDE','FGO_entry','FGO_syn','FGE_entry','FGE_syn'], 'SD(cd)+FG'),
        #                 (['SDO','SDE','FO_entry','FO_syn','FE_entry','FE_syn'],'SD(cd)+F'),
        #                 (['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn'],'SD(cd)+IC(n)+IC(OP)'),
        #                 (['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],'SD(cd)+IC(n)+IC(OP)+FG'),
        #                 (['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],'SD(cd)+IC(n)+IC(OP)+F'),
        #                 ]
        
        features_sets = [(['SDO','SDE'],'SD (cd)'),
                        (['SDO','SDE','DD_N_entry','DD_N_syn'], 'SD(cd)+DD(n)'),
                        (['SDO','SDE','DD_OP_entry','DD_OP_syn'],'SD(cd)+DD(OP)'),
                        (['SDO','SDE','FGO_entry','FGO_syn','FGE_entry','FGE_syn'], 'SD(cd)+FG'),
                        (['SDO','SDE','FO_entry','FO_syn','FE_entry','FE_syn'],'SD(cd)+F'),
                        (['SDO','SDE','DD_N_entry','DD_N_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],'SD(cd)+DD(n)+FG'),
                        (['SDO','SDE','DD_N_entry','DD_N_syn','FO_entry','FO_syn','FE_entry','FE_syn'],'SD(cd)+DD(n)+F'),
                        (['SDO','SDE','DD_OP_entry','DD_OP_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],'SD(cd)+DD(OP)+FG'),
                        (['SDO','SDE','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],'SD(cd)+DD(OP)+F'),
                        # (['SDO','SDE','DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],'SD(cd)+DD+F'),
                        ]
        
        for features_set, set_name in features_sets:
            clf = SynonymyClassif(
                            clf_model = LogisticRegression,
                            features = features_set,
                            kwargs_dict = {'class_weight':'balanced','max_iter':1500, 'penalty':'none'},
                            preprocessor=StandardScaler()
                            )
            clf.fit(X_train,y_train)
            this_name = name + set_name
            scores_per_model[this_name], scores_names = clf.score(X_test,y_test)
        
        #===== Logistic Regression SD(nk)=====
        name = 'LR '
        for k in SND_K_RANGE:
            # features_sets = [(['SNDO_'+str(k),'SNDE_'+str(k)],f'SD(n{k})'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn'], f'SD(n{k})+IC(n)'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'DD_OP_entry','DD_OP_syn'],f'SD(n{k})+IC(OP)'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'FO_entry','FO_syn','FE_entry','FE_syn'], f'SD(n{k})+FG'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'FO_entry','FO_syn','FE_entry','FE_syn'],f'SD(n{k})+F'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn'],f'SD(n{k})+IC(n)+IC(OP)'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],f'SD(n{k})+IC(n)+IC(OP)+FG'),
            #                 (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],f'SD(n{k})+IC(n)+IC(OP)+F'),
            #                 ]
            
            features_sets = [(['SNDO_'+str(k),'SNDE_'+str(k)],f'SD(n{k})'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn'], f'SD(n{k})+DD(n)'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_OP_entry','DD_OP_syn'],f'SD(n{k})+DD(OP)'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'FGO_entry','FGO_syn','FGE_entry','FGE_syn'], f'SD(n{k})+FG'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'FO_entry','FO_syn','FE_entry','FE_syn'],f'SD(n{k})+F'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],f'SD(n{k})+DD(n)+FG'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','FO_entry','FO_syn','FE_entry','FE_syn'],f'SD(n{k})+DD(n)+F'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_OP_entry','DD_OP_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],f'SD(n{k})+DD(OP)+FG'),
                        (['SNDO_'+str(k),'SNDE_'+str(k),'DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn'],f'SD(n{k})+DD(OP)+F'),
                        # (['SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],f'SD(n{k})+DD+F'),
                        ]
        
            
            for features_set, set_name in features_sets:
                clf = SynonymyClassif(
                                clf_model = LogisticRegression,
                                features = features_set,
                                kwargs_dict = {'class_weight':'balanced','max_iter':1500, 'penalty':'none'},
                                preprocessor=StandardScaler()
                                )
                clf.fit(X_train,y_train)
                this_name = name + set_name
                scores_per_model[this_name], scores_names = clf.score(X_test,y_test)
        
        #===== Logistic Regression multi =====
        name = 'LR '
        for k in SND_K_RANGE:
            clf = SynonymyClassif(
                            clf_model = LogisticRegression,
                            features = ['SDO','SDE','SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn'],
                            kwargs_dict = {'class_weight':'balanced','max_iter':1500, 'penalty':'none'},
                            preprocessor=StandardScaler()
                            )
            clf.fit(X_train,y_train)
            this_name = name + f'SD(cd+n{k})+DD+F'
            scores_per_model[this_name], scores_names = clf.score(X_test,y_test)
    
        

        # #===== Logistic Regression =====
        # # name = 'L1 LR'
        # # clf = SynonymyClassif(
        # #                 clf_model = LogisticRegression,
        # #                 features = features,
        # #                 kwargs_dict = {'class_weight':['balanced'],'max_iter':[500], 'penalty':['l1'], 'solver':['liblinear'],'C':(np.arange(1,41)/20)},
        # #                 CVgridSearch = True,
        # #                 CVgridSearch_score = make_scorer(matthews_corrcoef),
        # #                 standardscale=True
        # #                 )
        # # clf.fit(X_train,y_train)
        # # scores_per_model[name], scores_names = clf.score(X_test,y_test)
        # # print(f"Best C for regul LogReg on {synsSet} : {clf.model.best_params_['C']}")
        # # # print('-------- Coeff L1 LR -----------')
        # # print(pd.Series(index=features,data=clf.model.best_estimator_.coef_[0].round(3)).sort_values())
        
        # # lr_features = pd.DataFrame(data=clf.model.best_estimator_.coef_[0].round(2) , columns=['L1'], index=features)
        

        # # #===== Feature-selected Logistic Regression =====
        # # name = 'AIC LR'
        # # clf = SynonymyClassif(
        # #                 clf_model = LogisticRegression,
        # #                 features = features,
        # #                 kwargs_dict = {'class_weight':'balanced','max_iter':500, 'penalty':'none'},
        # #                 standardscale=True
        # #                 )
        # # clf.fit(X_train,y_train)
        # # clf.logreg_feature_selection(X_train,y_train,aic)
        # # scores_per_model[name], scores_names = clf.score(X_test,y_test)
        # # # print('-------- Coeff AIC LR -----------')
        # # # print(pd.Series(index=clf.features,data=clf.model.coef_[0].round(3)).sort_values())

        # # lr_features['AIC'] = pd.Series(index=clf.features,data=clf.model.coef_[0].round(2)).sort_values()
        # # lr_features.fillna(0, inplace=True)
        # # lr_features.mask((lr_features < 0.05) & (lr_features > -0.05), 0, inplace=True)
        # # lr_features_coeff_per_synSets[synsSet] = lr_features

        for k in SND_K_RANGE:

            features_set = ['SDO','SDE','SNDO_'+str(k),'SNDE_'+str(k),'DD_N_entry','DD_N_syn','DD_OP_entry','DD_OP_syn','FO_entry','FO_syn','FE_entry','FE_syn','FGO_entry','FGO_syn','FGE_entry','FGE_syn']
            #===== LR multi poly ======
            name = f'LR multi. poly. (2) (n{k})'
            clf = SynonymyClassif(
                            clf_model = LogisticRegression,
                            features = features_set,
                            kwargs_dict = {'class_weight':'balanced','max_iter':1500, 'penalty':'none'},
                            preprocessor = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2))
                            )
            clf.fit(X_train,y_train)
            scores_per_model[name], _ = clf.score(X_test,y_test)
            
            #===== SVM ======
            name = f'SVM (gaussian) (n{k})'
            clf = SynonymyClassif(
                            clf_model = SVC,
                            features = features_set,
                            kwargs_dict = {'kernel':['rbf'], 'gamma':['scale'], 'class_weight' :['balanced'],'C':(np.arange(1,41)/20)},
                            CVgridSearch = True,
                            CVgridSearch_score = make_scorer(balanced_accuracy_score),
                            preprocessor=StandardScaler()
                            )
            clf.fit(X_train,y_train)
            scores_per_model[name], _ = clf.score(X_test,y_test)
            
        #===== Freq LR ======
        name = 'LR Freq (F)'
        clf = SynonymyClassif(
                        clf_model = LogisticRegression,
                        features = ['FO_entry','FO_syn','FE_entry','FE_syn'],
                        kwargs_dict = {'class_weight':'balanced','max_iter':1000, 'penalty':'none'},
                        preprocessor = StandardScaler()
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)
        
        name = 'LR Freq (FG)'
        clf = SynonymyClassif(
                        clf_model = LogisticRegression,
                        features = ['FGO_entry','FGO_syn','FGE_entry','FGE_syn'],
                        kwargs_dict = {'class_weight':'balanced','max_iter':1000, 'penalty':'none'},
                        preprocessor = StandardScaler()
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)
        
        # #===== Divergence-and-Freq ======
        # for k in SND_K_RANGE:
        #     name = f'Div\\&Freq (n{k})'
        #     clf = SynonymyClassif(
        #                     clf_model = DivAndFreqCombinedClassifier,
        #                     features = features,
        #                     kwargs_dict = {'SD_feature':'NDiv_'+str(k), 'div_threshold':None, 'freq_threshold':0}
        #                     )
        #     clf.fit(X_train,y_train)
        #     scores_per_model[name], _ = clf.score(X_test,y_test)
        
        #===== Xu Kemp Control pairs
        name = f'XK controls'
        clf = SynonymyClassif(
                        clf_model = PreComputedClassifier,
                        features = features,
                        kwargs_dict = {'SD_feature':'XKcontrols'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)
        

        #===== Divergence-based======
        for k in SND_K_RANGE:
            name = f'Div (n{k})'
            clf = SynonymyClassif(
                            clf_model = PreComputedClassifier,
                            features = features,
                            kwargs_dict = {'SD_feature':'NDivG_avg_'+str(k)}
                            )
            clf.fit(X_train,y_train)
            scores_per_model[name], _ = clf.score(X_test,y_test)

        name = f'Div (cd)'
        clf = SynonymyClassif(
                        clf_model = PreComputedClassifier,
                        features = features,
                        kwargs_dict = {'SD_feature':'DivG_avg'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)

        name = f'Div (cd) Close'
        clf = SynonymyClassif(
                        clf_model = PreComputedClassifier,
                        features = features,
                        kwargs_dict = {'SD_feature':'DivG_closeavg'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)

        #===== Divergence-based SUPERVISED======
        for k in SND_K_RANGE:
            name = f'Div (n{k}) (tuned)'
            clf = SynonymyClassif(
                            clf_model = DivergenceBasedClassifier,
                            features = features,
                            kwargs_dict = {'SD_feature':'NDiv_'+str(k)}
                            )
            clf.fit(X_train,y_train)
            scores_per_model[name], _ = clf.score(X_test,y_test)

        name = f'Div (cd) (tuned)'
        clf = SynonymyClassif(
                        clf_model = DivergenceBasedClassifier,
                        features = features,
                        kwargs_dict = {'SD_feature':'Div'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)

        # #===== Frequence-based ======
        # name = 'Freq'
        # clf = SynonymyClassif(
        #                 clf_model = FrequencyBasedClassifier,
        #                 features = features,
        #                 kwargs_dict = dict()
        #                 )
        # clf.fit(X_train,y_train)
        # scores_per_model[name], _ = clf.score(X_test,y_test)

        #===== Always True ======
        name = 'AllTrue'
        clf = SynonymyClassif(
                        clf_model = ConstantOutputClassifier,
                        features = features,
                        kwargs_dict = {'constant':1,'out_type':'bool'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)

        #===== Always False ======
        name = 'AllFalse'
        clf = SynonymyClassif(
                        clf_model = ConstantOutputClassifier,
                        features = features,
                        kwargs_dict = {'constant':0,'out_type':'bool'}
                        )
        clf.fit(X_train,y_train)
        scores_per_model[name], _ = clf.score(X_test,y_test)

        #====== Summary ======
        scores_dataFrames_per_synset[synsSet] += [pd.DataFrame(data=scores_per_model,index=scores_names).transpose()]


summary_df_per_synset = dict()
for dataname, df_list in scores_dataFrames_per_synset.items():
    df = df_list[0]
    for df2 in df_list[1:]:
        df += df2
    df = df / len(df_list)
    summary_df_per_synset[dataname]=df

for synsSet in scores_dataFrames_per_synset.keys():
    print('=========== ' + synsSet + ' ===========')
    print('-------- Performances -----------')
    display(summary_df_per_synset[synsSet])

print('\n======= AS LATEX =======\n')

print("\\begin{tabular}{c|cccc}\\toprule")
print(pos.upper()  + " & " + ' & '.join(scores_names) + " \\\\" )
for row_i, row in summary_df_per_synset[synsSet][['PD','F1 (syns)','F1 (diff)','BA']].round(2).iterrows():
    print(str(row_i) + ' & ' + " & ".join(list(row.values.astype('str'))) + ' \\\\')
print("\\bottomrule\n\\end{tabular}")


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
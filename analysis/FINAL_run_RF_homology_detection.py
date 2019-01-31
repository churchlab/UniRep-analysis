#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:06:41 2018

@author: gr
"""

import time

start = time.time()

import numpy as np
import pandas as pd
import sys
sys.path.append('../')

import common_v2.validation_tools as validation_tools
from sklearn.externals import joblib
import os

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, explained_variance_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier
import sklearn
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from common_v2.validation_tools import subsets, regr_datasets, reps, none_model_reps, auc_datasets, get_tvt, get_roc

import random

reps = ["avg_hidden"]

num_bayes_it = 75

to_train = True

random.seed(42)
np.random.seed(42)

path_to_pieces = f"../../data/pieces_new/"
run_type='validate'

merged_paths = {
        "handstad_superfamily":"../../data/Handstad_2007_remote_homology_SCOP_1.67/Handstad_2007_superfamilies_merged.csv",
        "handstad_folds":"../../data/Handstad_2007_remote_homology_SCOP_1.67/Handstad_2007_folds_merged.csv"}

merged_groups = {
        "handstad_superfamily":['a.1.1.2', 'a.1.1.3', 'a.3.1.1', 'a.4.1.1', 'a.22.1.1', 'a.22.1.3', 'a.25.1.1',
                                'a.25.1.2', 'a.26.1.1', 'a.26.1.2', 'a.26.1.3', 'a.39.1.2', 'a.39.1.5',
                                'a.39.1.8', 'a.138.1.1', 'a.138.1.3', 'b.1.1.1', 'b.1.1.2', 'b.1.1.3',
                                'b.1.1.4', 'b.1.18.2', 'b.6.1.1', 'b.6.1.3', 'b.121.4.1', 'b.121.4.7',
                                'b.29.1.1', 'b.29.1.2', 'b.29.1.3', 'b.29.1.11', 'b.40.2.1', 'b.40.2.2',
                                'b.40.4.3', 'b.40.4.5', 'b.47.1.1', 'b.47.1.2', 'b.47.1.3', 'b.50.1.1',
                                'b.50.1.2', 'b.55.1.1', 'b.60.1.1', 'b.60.1.2', 'b.82.1.2', 'c.1.2.4',
                                'c.1.8.1', 'c.1.8.3', 'c.1.8.4', 'c.1.8.5', 'c.1.10.1', 'c.2.1.1', 'c.2.1.2',
                                'c.2.1.3', 'c.2.1.4', 'c.2.1.5', 'c.2.1.6', 'c.2.1.7', 'c.3.1.2', 'c.3.1.5',
                                'c.26.1.1', 'c.26.1.3', 'c.37.1.1', 'c.37.1.8', 'c.37.1.9', 'c.37.1.10',
                                'c.37.1.11', 'c.37.1.12', 'c.37.1.19', 'c.37.1.20', 'c.45.1.2', 'c.47.1.1',
                                'c.47.1.5', 'c.47.1.10', 'c.55.1.1', 'c.55.1.3', 'c.55.3.1', 'c.55.3.5',
                                'c.56.5.4', 'c.67.1.1', 'c.67.1.3', 'c.67.1.4', 'c.94.1.1', 'c.94.1.2',
                                'd.2.1.2', 'd.2.1.3', 'd.3.1.1', 'd.15.1.1', 'd.15.4.1', 'd.15.4.2',
                                'd.32.1.3', 'd.81.1.1', 'd.92.1.11', 'd.108.1.1', 'd.153.1.4', 'd.169.1.1',
                                'g.3.6.1', 'g.3.6.2', 'g.3.7.1', 'g.3.7.2', 'g.3.11.1', 'g.14.1.1', 'g.14.1.2',
                                'g.39.1.2', 'g.39.1.3'],
        "handstad_folds":['a.2.11', 'a.4.1', 'a.4.5', 'a.4.6', 'a.5.2', 'a.7.1', 'a.60.1', 'a.102.1', 'a.102.4',
                          'a.118.1', 'a.118.8', 'b.1.1', 'b.1.18', 'b.1.2', 'b.1.6', 'b.1.8', 'b.2.2', 'b.2.3',
                          'b.2.5', 'b.121.4', 'b.121.5', 'b.34.2', 'b.34.5', 'b.40.2', 'b.40.4', 'b.40.6',
                          'b.42.1', 'b.42.2', 'b.43.4', 'b.43.3', 'b.68.1', 'b.69.4', 'b.80.1', 'b.82.1',
                          'b.82.2', 'b.82.3', 'b.84.1', 'c.1.1', 'c.1.2', 'c.1.4', 'c.1.7', 'c.1.8', 'c.1.9',
                          'c.1.10', 'c.1.11', 'c.1.12', 'c.1.15', 'c.23.1', 'c.23.5', 'c.23.12', 'c.23.16',
                          'c.26.1', 'c.26.2', 'c.55.1', 'c.55.3', 'c.56.2', 'c.56.5', 'd.15.1', 'd.15.2',
                          'd.15.4', 'd.15.6', 'd.15.7', 'd.17.2', 'd.17.4', 'd.26.1', 'd.26.3', 'd.58.1',
                          'd.58.3', 'd.58.5', 'd.58.7', 'd.58.17', 'd.58.26', 'd.79.1', 'd.110.3', 'd.129.1',
                          'd.129.3', 'f.1.4', 'g.3.1', 'g.3.2', 'g.3.6', 'g.3.7', 'g.3.9', 'g.3.11', 'g.3.13',
                          'g.41.3', 'g.41.5']
}



def get_remote_homology_split(merged_df, family_name):

    test_positive =  merged_df[merged_df[family_name] == 1][['rep','family_name']] #1
    test_negative =  merged_df[merged_df[family_name] == 2][['rep','family_name']] #2
    train_positive = merged_df[merged_df[family_name] == 3][['rep','family_name']] #3
    train_negative = merged_df[merged_df[family_name] == 4][['rep','family_name']] #4

    test_positive.columns = ['rep','target']
    test_negative.columns = ['rep','target']
    train_positive.columns = ['rep','target']
    train_negative.columns = ['rep','target']

    test_positive.loc[:,'target'] = 1
    test_negative.loc[:,'target'] = 0
    train_positive.loc[:,'target'] = 1
    train_negative.loc[:,'target'] = 0

    test_positive.loc[:,'target'] = 1
    test_negative.loc[:,'target'] = 0

    train_positive.loc[:,'target'] = 1
    train_negative.loc[:,'target'] = 0

    test = pd.concat([test_negative, test_positive]).sample(frac=1)
    train = pd.concat([train_negative, train_positive]).sample(frac=1)

    return train, test

def get_merged_df(d, rep, path):

    all_rep, v, t = validation_tools.get_tvt(path,
                                         d,
                                         'full',
                                         rep,
                                         dataset_filetype='pkl',
                                         verbose=True)
    #scaled automatically

    all_seq, v, t = validation_tools.get_tvt(path,
                                         d,
                                         'full',
                                         'sequence',
                                          modifiers=[],
                                         dataset_filetype='pkl',
                                         verbose=False,
                                         )

    assert (all_seq.target == all_rep.target).all()

    all_seq.columns = ['seq','family_name']

    sf = pd.concat([all_rep,all_seq],axis=1)[['seq','rep','family_name']]

    merged_df = pd.read_csv(merged_paths[d], index_col='seq')

    sf = sf.set_index('seq')

    merged_df = merged_df.loc[sf.index] # THIS IS TO DROP RGN STRUGGLERS

    merged_df = sf.join(merged_df)
    merged_df.index.name = 'seq'
    merged_df = merged_df.reset_index()
    return merged_df

from skopt.space import Real, Categorical, Integer

models={
                        'RF_informedguess':{ 'param_grid':{
                                    'n_estimators':[1000],
                                    'max_features':Real(0.5,1.0),
                            		'n_jobs':[-1],
                            		'class_weight':['balanced'],
                                    'criterion':['gini','entropy'],
                                    'min_samples_leaf': Real(0.1,0.3),
                                    'min_samples_split':Real(0.01,1.0),
                                    },
                                    'model':RandomForestClassifier
                            }
                    }

if to_train:
    model_list = models.keys()
else:
    model_list = ['RF_informedguess']

for model_name in model_list:
    print(model_name)
    m_results={}
    m_params={}

    for d in auc_datasets:
        print(d)

        d_results = {}
        d_params={}

        for rep in reps:
            print(rep)
            merged_df = get_merged_df(d, rep, os.path.join(path_to_pieces, f"{d}__full.pkl"))

            r_results = {}
            r_params = {}

            for fam_name in merged_groups[d]:
                print(fam_name)
                train, validate = get_remote_homology_split(merged_df, fam_name)

                try:
                    if to_train:
                        model=BayesSearchCV(
                                        models[model_name]['model'](),
                                        models[model_name]['param_grid'],
                                        n_iter=num_bayes_it,
                                        scoring='roc_auc',
                                        cv=3,
                                        n_jobs=-1
                                    )


                        X = np.asarray(train['rep'].values.tolist())
                        y = train['target'].values.astype('float64')

                        model.fit(
                                       X ,
                                        y

                        )
                        joblib.dump(model, f'./models/auc__{d}__{rep}__{fam_name}__model.pkl')
                    else:
                        model = joblib.load(f'./models/auc__{d}__{rep}__{fam_name}__model.pkl')

                    predictions = model.predict(

                                    np.asarray(validate['rep'].values.tolist()),

                    )

                    r_results[fam_name] = {
                            'roc_score':roc_auc_score(validate['target'], predictions),
                            'roc50_score':get_roc(validate['target'].values, predictions, 50)
                        }
                    print('roc_score', r_results[fam_name]['roc_score'])
                    print('roc50_score', r_results[fam_name]['roc50_score'])
                    if to_train:
                        if model_name != 'NaiveBayes':
                            r_params[fam_name] = model.best_params_
                            print(model.best_params_)
#
                except Exception as e:
                    print(f"{d} {rep} {fam_name} ERROR {e}")
                    r_results[fam_name] = {
                        'roc_score':np.nan,
                        'roc50_score':np.nan
                    }

                end = time.time()
                print(end - start)

            d_results[rep] = r_results
            d_params[rep] = r_params
            joblib.dump(r_results, f"./results/auc__{model_name}__{d}__{rep}__results.pkl")
            if to_train:
                joblib.dump(r_params, f"./params/auc__{model_name}__{d}__{rep}__best_params.pkl")

        m_results[d] = d_results
        m_params[d] = d_params

    joblib.dump(m_results, f"./results/auc__{model_name}__all__results.pkl")
    if to_train:
        joblib.dump(m_params, f"./params/auc__{model_name}__all__best_params.pkl")

    end = time.time()
    print(f"overall time for {model_name} {d}", end - start)

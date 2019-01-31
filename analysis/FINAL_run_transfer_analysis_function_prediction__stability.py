#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#####
# THIS RUNS ALL VS ALL SUBSET TRANSFER
#####
Created on Sun Aug 19 15:36:27 2018
@author: gr
"""

import pandas as pd
import numpy as np

from sklearn.externals import joblib
import os
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("../")

from common_v2.validation_tools import subsets, NullModel, pearson, transfer_datasets, trainable_transfer_mean_model, tvt_modifier_baseline_reps, tvt_modifier_return_mean, reps
import random
import common_v2.validation_tools
import common_v2.validation_tools as validation_tools

from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection import KFold

random.seed(42)
np.random.seed(42)


baseline_rep = 'simple_freq_and_len' #  DO NOT CHANGE WITHOUT THINKING
#  THROUGH THE CONSEQUENCES, potential for hard-to-figure-out bugs
#  Basically, need to make sure ngrams are constructed on the correct train and
#  then used on the correct validate/test set for each of the models

transfer_datasets = [
 'fowler_consistent_single_UBI',
 'rocklin_ssm2_nat_eng',
 'rocklin_ssm2',
                     'rocklin_ssm2_remote_test']

path_to_pieces = f"../../data/pieces_new/"

to_train = True

metrics = {
            'error':mean_squared_error,
            'pearson':pearson
            }

# if training, just record validation results (to avoid retraining). if not training, record all results
if to_train:
    run_types=['validate']
else:
     run_types = ['train','validate','test']

for run_type in run_types:
    print(run_type)
    for d in transfer_datasets:
        print(d)

        if d == 'rocklin_ssm2_nat_eng':
            preloaded_dataset = pd.read_pickle(
                        os.path.join(
                                path_to_pieces,f"{'__'.join(['rocklin_ssm2','full'])}.pkl"
                            )
                    )
            nat_ind = preloaded_dataset[
                    ~preloaded_dataset.phenotype_name.str.contains(".pdb")
                ].index.copy()
            eng_ind = preloaded_dataset[
                    preloaded_dataset.phenotype_name.str.contains(".pdb")
                ].index.copy()
            print('preloaded_dataset.shape', preloaded_dataset.shape)

            preloaded_dataset_init = preloaded_dataset.copy()

            ###

            preloaded_dataset.loc[nat_ind, 'phenotype_name'] = 'natural'
            preloaded_dataset.loc[eng_ind, 'phenotype_name'] = 'engineered'

            print(preloaded_dataset.phenotype_name.unique())

        elif d == 'fowler_consistent_single_UBI':
            preloaded_dataset = pd.read_pickle(
                        os.path.join(path_to_pieces,f"{'__'.join(['fowler','full'])}.pkl")
                    )
            ubi_ind = preloaded_dataset[
                    (preloaded_dataset.phenotype_name == 'Ubiquitin') | (preloaded_dataset.phenotype_name == 'E1_Ubiquitin')].index.copy()
            preloaded_dataset.loc[ubi_ind, 'phenotype_name'] = 'UBI4_combined'


        elif d == 'rocklin_ssm2_remote_test':

            remote_names = ["EEHEE_rd3_0037.pdb_ssm2_stability", "HHH_rd2_0134.pdb_ssm2_stability",
                "HHH_rd3_0138.pdb_ssm2_stability", "villin_ssm2_stability",
                "EHEE_rd2_0005.pdb_ssm2_stability", "hYAP65_ssm2_stability",
                "HEEH_rd3_0872.pdb_ssm2_stability", "EEHEE_rd3_1498.pdb_ssm2_stability"]

            preloaded_dataset = pd.read_pickle(
                        os.path.join(
                                path_to_pieces,f"{'__'.join(['rocklin_ssm2','full'])}.pkl"
                            )
                    )
            remote_ind = preloaded_dataset[
                    preloaded_dataset.phenotype_name.isin(remote_names)
                ].index.copy()
            central_ind = preloaded_dataset[
                    ~preloaded_dataset.phenotype_name.isin(remote_names)
                ].index.copy()
            print(central_ind.shape)
            print('preloaded_dataset.shape', preloaded_dataset.shape)

            preloaded_dataset_init = preloaded_dataset.copy()

            ###

            preloaded_dataset.loc[remote_ind, 'phenotype_name'] = 'remote'
            preloaded_dataset.loc[central_ind, 'phenotype_name'] = 'central'
            print(preloaded_dataset.phenotype_name.value_counts())
            print(preloaded_dataset.phenotype_name.unique())

        else:
            preloaded_dataset = pd.read_pickle(
                        os.path.join(path_to_pieces,f"{'__'.join([d,'full'])}.pkl")
                    )
        print('preloaded dataset')



        for rep in reps:
            print("    "+rep)
            ss = subsets[d] #  subsets for this dataset (precomputed and saved in
            #  advance, need to updated if there is a new dataset file)

            results_eval_on_holdout = pd.DataFrame(columns= ['transfer_ratio_avg', 'indomain_ratio_avg'])

            errors_by_s = pd.DataFrame(
                    index=ss,
                    columns=list(np.array(
                            [[f'{k}_S_T', f'{k}_T_T', f'b_{k}_T_T'] for k in metrics.keys()]
                        ).flatten()) + ['transfer_ratio','indomain_ratio']
                )

            for s in ss:

                #  SOURCE TRAIN, VALIDATE/TEST (all but one s)

                print("        "+f"Preparing source splits: dataset {d}, all but {s}")

                source_train, source_validate, source_test = validation_tools.get_tvt(
                        preloaded_dataset,#os.path.join(path_to_pieces, name_of_full_dataset),
                        d,
                        s,
                        rep,
                        dataset_filetype='loaded_full_dataset', #'pkl',
                        verbose=False,
                        all_but_one_phenotype=True,
                        is_transfer=True,
                        modifiers=[tvt_modifier_baseline_reps,
                                                                        tvt_modifier_return_mean]
                        )
                X_source_train = np.asarray(source_train['rep'].values.tolist())
                y_source_train = source_train['target'].values


                if rep != 'mean':
                    print('        creating lopo split')
                    logo = LeaveOneGroupOut()
                    if (d != 'rocklin_ssm2_nat_eng') & (d != 'rocklin_ssm2_remote_test'):
                        groups_source_train, kk, k = validation_tools.get_tvt(
                                preloaded_dataset, 
                                d,
                                s,
                                'phenotype_name', # to split by phenotype name
                                dataset_filetype='loaded_full_dataset',
                                verbose=False,
                                all_but_one_phenotype=True,
                                is_transfer=True,
                                modifiers=[])

                        groups_source_train = groups_source_train.rep.tolist()
                    else:
                        seqs_source_train, kk, k = validation_tools.get_tvt(
                                preloaded_dataset,
                                d,
                                s,
                                'sequence',
                                dataset_filetype='loaded_full_dataset',
                                verbose=False,
                                all_but_one_phenotype=True,
                                is_transfer=True,
                                modifiers=[])

                        groups_source_train = preloaded_dataset_init.set_index('sequence').loc[seqs_source_train['rep'], 'phenotype_name'].tolist()
                        print("created special rocklin lopo split")

                    splits_source_train = [
                            (tr,te) for tr, te in logo.split(
                                    X_source_train, y_source_train, groups=groups_source_train
                                )
                        ]


                #  TARGET TRAIN, VALIDATE/TEST (the one s)

                print("        "+f"Preparing target splits: dataset {d}, the subset {s}")

                name_of_subset = f"{'__'.join([d,s])}.pkl"

                target_train, target_validate, target_test = validation_tools.get_tvt(
                        preloaded_dataset,
                        d,
                        s,
                        rep,
                        dataset_filetype='loaded_full_dataset',
                        verbose=False,
                        all_but_one_phenotype=False,
                        is_transfer=True,
                        modifiers=[tvt_modifier_baseline_reps,
                                                                        tvt_modifier_return_mean]
                        )

                print(f"        Preparing baseline splits: dataset {d}, the subset {s}")

                baseline_target_train, baseline_target_validate, baseline_target_test = validation_tools.get_tvt(
                        preloaded_dataset,
                        d,
                        s,
                        baseline_rep,
                        dataset_filetype='loaded_full_dataset',
                        verbose=False,
                        all_but_one_phenotype=False,
                        is_transfer=False,
                        modifiers=[tvt_modifier_baseline_reps,
                                                                        tvt_modifier_return_mean]
                        #  no need for transfer logic for baseline -
                                          #  situation here  is normal regression wrt baseline
                        )

                print(f"        Prepared splits for {d} held out {s}")


                if rep == 'mean': # no need for cross-validation here
                    model_trained_on_S = trainable_transfer_mean_model()
                    model_trained_on_S.fit(X_source_train, y_source_train)

                    model_trained_on_T = trainable_transfer_mean_model()
                    model_trained_on_T.fit(np.asarray(target_train['rep'].values.tolist()), target_train['target'].values)

                    baseline_trained_on_T = LassoLarsCV(fit_intercept = True,
                                                normalize = True,
                                                n_jobs=-1,
                                                max_n_alphas=6000,
                                                cv=KFold(n_splits=10,
                                                         random_state=42, shuffle=True)
                                        ).fit(np.asarray(baseline_target_train['rep'].values.tolist()), baseline_target_train['target'].values)

                else:
                    if to_train:

                        model_trained_on_S = LassoLarsCV(
                                                fit_intercept = True,
                                                normalize = True,
                                                n_jobs=-1,
                                                cv=splits_source_train,
                                                max_n_alphas=6000,
                                            ).fit(X_source_train, y_source_train)

                        model_trained_on_T = LassoLarsCV(
                                                fit_intercept = True,
                                                normalize = True,
                                                n_jobs=-1,
                                                max_n_alphas=6000,
                                                cv=KFold(n_splits=10, random_state=42, shuffle=True)
                                            ).fit(np.asarray(target_train['rep'].values.tolist()), target_train['target'].values)

                        baseline_trained_on_T = LassoLarsCV(
                                                fit_intercept = True,
                                                normalize = True,
                                                n_jobs=-1,
                                                max_n_alphas=6000,
                                                cv=KFold(n_splits=10, random_state=42, shuffle=True)
                                            ).fit(np.asarray(baseline_target_train['rep'].values.tolist()),
                                                             baseline_target_train['target'].values)
                        joblib.dump(model_trained_on_S,    f"./models/transfer_holdout__{d}__{s}__{rep}_model_trained_on_S.mkl")
                        joblib.dump(model_trained_on_T,    f"./models/transfer_holdout__{d}__{s}__{rep}_model_trained_on_T.mkl")
                        joblib.dump(baseline_trained_on_T, f"./models/transfer_holdout__{d}__{s}__{rep}_baseline_trained_on_T.mkl")
                    else:
                        model_trained_on_S = joblib.load(
                                f"./models/transfer_holdout__{d}__{s}__{rep}_model_trained_on_S.mkl")

                        model_trained_on_T = joblib.load(
                                f"./models/transfer_holdout__{d}__{s}__{rep}_model_trained_on_T.mkl")

                        baseline_trained_on_T = joblib.load(
                                f"./models/transfer_holdout__{d}__{s}__{rep}_baseline_trained_on_T.mkl")

                errors = pd.Series()

                if run_type == 'train':
                    target_validate = target_train.copy()
                    baseline_target_validate = baseline_target_train.copy()

                if run_type == 'test':
                    target_validate = target_test.copy()
                    baseline_target_validate = baseline_target_test.copy()

                for key in metrics.keys():
                # Transfer Error
                    X = np.asarray(target_validate['rep'].values.tolist())
                    yhat = model_trained_on_S.predict(X)
                    np.save(f"./predictions/transfer__{d}__{s}__{rep}__{run_type}__predictions_S_T.npy", yhat)
                    errors.loc[f'{key}_S_T'] = metrics[key](target_validate['target'], yhat)

                    # In-domain Error
                    X = np.asarray(target_validate['rep'].values.tolist())
                    yhat = model_trained_on_T.predict(X)
                    np.save(f"./predictions/transfer__{d}__{s}__{rep}__{run_type}__predictions_T_T.npy", yhat)
                    errors.loc[f'{key}_T_T'] = metrics[key](target_validate['target'], yhat)

                    # Baseline in-domain error
                    X = np.asarray(baseline_target_validate['rep'].values.tolist())
                    yhat = baseline_trained_on_T.predict(X)
                    np.save(f"./predictions/transfer__{d}__{s}__{rep}__{run_type}__b_predictions_T_T.npy", yhat)
                    errors.loc[f'b_{key}_T_T'] = metrics[key](baseline_target_validate['target'], yhat)

                errors.loc['transfer_ratio']  = errors.loc['error_S_T'] / float(errors.loc['b_error_T_T'])
                errors.loc['indomain_ratio'] = errors.loc['error_T_T'] / float(errors.loc['b_error_T_T'])

                errors_by_s.loc[s,:] = errors.values
                print("        errors ready\n")
                #print(errors.values)

            errors_by_s.to_csv(f"./results/transfer__{d}__{rep}__{run_type}__metrics.csv")

            results_eval_on_holdout.loc[rep,:] = errors_by_s[['transfer_ratio', 'indomain_ratio']].mean().values
            results_eval_on_holdout.to_csv(f"./results/transfer__{d}__{rep}__{run_type}__results_eval_on_holdout.csv")

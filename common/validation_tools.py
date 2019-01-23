# -*- coding: utf-8 -*-
""" Module with functions for the validate pipeline

Created on Sat Aug 18 09:10:53 2018

@author: gr
"""

import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
import os
import random

random.seed(42)
np.random.seed(42)

from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, explained_variance_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

# To allow imports from common directory

import common.vectorize_for_benchmarking_lib
from common.vectorize_for_benchmarking_lib import get_dumb_features, get_freq_and_len, get_bow_representations, transform_bow_to_tfidf

regr_datasets = pd.Series(['leuenberger', 'solubility',
       'arnold_T50', 'arnold_absorption', 'arnold_enantioselectivity',
       'arnold_localization',
       'fowler',
       'rocklin_ssm2', 'rocklin_all_rds'], name='dataset')
transfer_datasets = pd.Series(['fowler_consistent_single_UBI', 'rocklin_ssm2',  'rocklin_ssm2_nat_eng'], name='dataset')
auc_datasets = pd.Series(['handstad_superfamily', 'handstad_folds'], name='dataset')

#Excluded 'handpicked', 'oxbench', 'prenyltransferase', 'RABmark_sup', 'RABmark_twi'


from scipy.stats import pearsonr
# https://github.com/EducationalTestingService/skll/blob/master/skll/metrics.py
def pearson(y_true, y_pred):
    """
    Calculate Pearson product-moment correlation coefficient between ``y_true``
    and ``y_pred``.
    Parameters
    ----------
    y_true : array-like of float
        The true/actual/gold labels for the data.
    y_pred : array-like of float
        The predicted/observed labels for the data.
    Returns
    -------
    ret_score : float
        Pearson product-moment correlation coefficient if well-defined, else 0.0
    """
    try:
        ret_score = pearsonr(y_true, y_pred)[0]
    except Exception as e:
        print(f"ERROR {e}")
        print("can't calc pearson, probably mean is to blame - no var, undefined corr")
        return np.nan
    return ret_score

metrics = {
'mse':mean_squared_error,
'r2':r2_score,
'exp_var':explained_variance_score,
'pearson_r':pearson}

#  the below is aparently what one has to do to import a local file into a module
import pkgutil
subsets = pkgutil.get_data(__package__, 'subsets.pkl')
from io import BytesIO
subsets = pd.read_pickle(BytesIO(subsets))

none_model_reps = ['mean']

reps = ["mean",
        "simple_freq_plus","simple_freq_and_len",
        "tfidf_3grams", "3grams", "tfidf_2grams", "2grams",
            "RGN", "64_avg_hidden","64_final_hidden","64_final_cell",
            "256_avg_hidden", "256_final_cell",
            "avg_hidden", "final_hidden", "final_cell",
            "arnold_original_3_7", "arnold_scrambled_3_5", "arnold_random_3_7", "arnold_uniform_4_1",
            'all_64', 'all_256',
       'all_1900', 'all_avg_hidden', 'all_final_cell', 'RGN_avg_hidden',
       'RGN_final_cell']

def make_name(dataset, phenotype_name, rep):
    return f"{dataset}__{phenotype_name}__{rep}"

is_baseline_rep = lambda rep: rep in ["simple_freq_plus","simple_freq_and_len",
                "tfidf_2grams", "2grams",
                "tfidf_3grams", "3grams"]

def tvt_modifier_baseline_reps(dataset, phenotype_name, rep, train, validate,
                               test, verbose,
                               all_but_one_phenotype,
                               is_transfer):
    """ Calculate baseline representation specified in rep, save relevant models,
    return modified train, validate, test
    """

    if is_baseline_rep(rep):

        # 'rep' column contains sequence in this case

        if rep == "simple_freq_plus":
            train['rep'] = get_dumb_features(train.rep)
            validate['rep'] = get_dumb_features(validate.rep)
            test['rep'] = get_dumb_features(test.rep)
            if verbose:
                print("Created simple simple_freq_plus reps")

        elif rep == "simple_freq_and_len":
            train['rep'] = get_freq_and_len(train.rep)
            validate['rep'] = get_freq_and_len(validate.rep)
            test['rep'] = get_freq_and_len(test.rep)
            if verbose:
                print("Created simple simple_freq_and_len reps")

        elif rep in [ "tfidf_2grams", "2grams",
                    "tfidf_3grams", "3grams"]:

            max_features = 10000

            if rep in [ "tfidf_2grams", "2grams"]:
                n = 2
            elif rep in [ "tfidf_3grams", "3grams"]:
                n = 3

            # for each ngramness, check if there is a saved vectorizer and transformer
            vectorizer_path = f'./models/{make_name(dataset, phenotype_name, rep)}_{n}_bow_vectorizer.pkl'
            transformer_path = f'./models/{make_name(dataset, phenotype_name, rep)}_{n}_tfidf_transformer.pkl'
            if is_transfer:
                vectorizer_path = f'./models/all_but_one_phenotype__{make_name(dataset, phenotype_name, rep)}_{n}_bow_vectorizer.pkl'
                transformer_path = f'./models/all_but_one_phenotype__{make_name(dataset, phenotype_name, rep)}_{n}_tfidf_transformer.pkl'
            paths_exists = os.path.exists(vectorizer_path) and os.path.exists(transformer_path)

            if paths_exists:
                if verbose:
                    print(f"Loading vectorizer ({vectorizer_path}) and transformer ({transformer_path})")
                vectorizer = joblib.load(vectorizer_path)
                transformer = joblib.load(transformer_path)

                train[f"{n}grams"], v = get_bow_representations(train.rep, vectorizer=vectorizer)
                train[f"tfidf_{n}grams"], t = transform_bow_to_tfidf(np.asarray(train[f"{n}grams"].values.tolist()),
                                                                    transformer=transformer)
            else:
                #  The below is a little tricky: basically in transfer, we want
                #  to only compute vectorizer on all_but_one dataset (source)
                #  and use those to compute ngram reps for the heldout
                #  dataset (target), while handling non-transfer behavior
                #  i.e. just compute new vectorizer and transformer if they
                #  are not found on disk
                if (not is_transfer) or (is_transfer and all_but_one_phenotype):
                    if verbose:
                        print(f"Creating new vectorizer ({vectorizer_path}) and transformer ({transformer_path})")
                    train[f"{n}grams"], vectorizer = get_bow_representations(train.rep, ngramness=n, max_features=max_features)
                    train[f"tfidf_{n}grams"], transformer = transform_bow_to_tfidf(np.asarray(train[f"{n}grams"].values.tolist()))
                else:
                    raise FileNotFoundError("Need all_but_one ngram vectorizer and transformer pre-computed to construct\
                                            transfer target dataset ngram reps")
            validate[f"{n}grams"], v = get_bow_representations(validate.rep, vectorizer=vectorizer)
            validate[f"tfidf_{n}grams"], t = transform_bow_to_tfidf(np.asarray(validate[f"{n}grams"].values.tolist()),
                                                                    transformer=transformer)

            test[f"{n}grams"], v = get_bow_representations(test.rep, vectorizer=vectorizer)
            test[f"tfidf_{n}grams"], t = transform_bow_to_tfidf(np.asarray(test[f"{n}grams"].values.tolist()),
                                                                    transformer=transformer)
            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(transformer, transformer_path)

            train['rep'] = train[rep]
            validate['rep'] = validate[rep]
            test['rep'] = test[rep]
    if verbose:
        print("train shape", train.shape)

    return train[['rep','target']], validate[['rep','target']], test[['rep','target']]

def tvt_modifier_scale_x(dataset, phenotype_name, rep, train, validate,
                         test, verbose,
                         all_but_one_phenotype,
                         is_transfer):
    """ Scales reps to mean 0 std 1 - trains scaler on train, transforms test
    and validate, handles transfer special case,
    returns modified train, validate, test"""

    if rep != 'mean': #  no need to transform the mean rep, there is no real
        #  model for it (the null model just returns rep)

        #  the below looks a little more complicated than it is because of
        #  transfer trickiness - basically, in transfer we want to train
        #  the scaler on all_but_one_phenotype and then use that to
        #  transform the train,validate,test of the heldout target phenotype

        #  here we try to load the scaler and transform train
        scaler_path = f'./models/{make_name(dataset, phenotype_name, rep)}_standard_scaler.pkl'
        if is_transfer:
            scaler_path = f'./models/all_but_one_phenotype__{make_name(dataset, phenotype_name, rep)}_standard_scaler.pkl'
        paths_exists = os.path.exists(scaler_path)

        if paths_exists:
            if verbose:
                print(f"Loading scaler ({scaler_path})")
            scaler = joblib.load(scaler_path)

            train['rep'] = scaler.transform(
                    np.asarray(train['rep'].values.tolist())
                ).tolist()

        #  if we can't load it we either create a new one or fail if loading is
        #  required (if we expect a trained scaler in transfer for target domain)
        else:

            if (not is_transfer) or (is_transfer and all_but_one_phenotype):
                if verbose:
                    print(f"Creating new scaler ({scaler_path})")

                scaler = StandardScaler()

                if verbose:
                    print("train rep shape", np.asarray(train['rep'].values.tolist()).shape)
                train['rep'] = scaler.fit_transform(
                        np.asarray(train['rep'].values.tolist())
                    ).tolist()

                joblib.dump(scaler, scaler_path) # saving for later

            else:
                raise FileNotFoundError("Need all_but_one scaler pre-computed to construct\
                                        transfer target dataset scaled {rep}")

        #  transforming non-trian splits
        validate['rep'] = scaler.transform(
                    np.asarray(validate['rep'].values.tolist())
                ).tolist()
        test['rep'] = scaler.transform(
                    np.asarray(test['rep'].values.tolist())
                ).tolist()

        if verbose:
            print(f"""
                  scaled reps
                  train.rep.mean:{np.asarray(train['rep'].values.tolist()).mean()}
                  validate.rep.mean:{np.asarray(validate['rep'].values.tolist()).mean()}

                  train.rep.std:{np.asarray(train['rep'].values.tolist()).std()}
                  validate.rep.std:{np.asarray(validate['rep'].values.tolist()).std()}""")


    return train, validate, test

def tvt_modifier_return_mean(dataset, phenotype_name, rep,
                                               train, validate, test, verbose,
                                               all_but_one_phenotype,
                                               is_transfer):
    """ Calculates mean for train, returns that mean
    as representation in train, validate and test"""

    if rep == "mean":

        #  Handling transfer weirdness - making sure to predict saved all_but_one mean
        #  on the heldout target dataset in transfer
        mean_path = f'./models/{make_name(dataset, phenotype_name, rep)}_mean.pkl'
        if is_transfer:
            mean_path = f'./models/all_but_one_phenotype__{make_name(dataset, phenotype_name, rep)}_mean.pkl'
        paths_exists = os.path.exists(mean_path)

        if paths_exists:
            mean = joblib.load(mean_path)
            if verbose: print(f"Loaded mean {mean_path}")

        else:
            if (not is_transfer) or (is_transfer and all_but_one_phenotype):
                mean = train.target.mean()
                joblib.dump(mean, mean_path)
                if verbose: print(f"Computed and saved a new mean: {mean}")
            else:
                raise FileNotFoundError("Need all_but_one mean pre-computed to construct\
                                        transfer target dataset mean")

        if verbose:
            print('putting mean as the rep: ', mean)

        train.loc[:,'rep'] = [mean] * train.shape[0]
        test.loc[:,'rep'] = [mean] * test.shape[0]
        validate.loc[:,'rep'] = [mean] * validate.shape[0]

        train.rep = train.rep.apply(lambda x: [x])
        test.rep = test.rep.apply(lambda x: [x])
        validate.rep = validate.rep.apply(lambda x: [x])

    return train, validate, test

def npy_to_df_old(file):
    """
    given a npy prefix, returns df with the right col_names
    """
    col_names = ["sequence", "phenotype", "is_train", "is_test",
                 "phenotype_name", "dataset", "RGN", "64_avg_hidden",
                 "64_final_hidden","64_final_cell", "256_avg_hidden",
                 "256_final_hidden", "256_final_cell", "avg_hidden",
                 "final_hidden", "final_cell","arnold_original_3_7",
                 "arnold_scrambled_3_5", "arnold_random_3_7",
                 "arnold_uniform_4_1"]
    return pd.DataFrame(np.load(file), columns=col_names)

def npy_to_df(file):
    """
    given a npy prefix, returns df with the right col_names for the concat file
    'final_with_concat.npy'
    which has new concatted reps and dropped 256_final_hidden which had a putative
    off-by-one error. Also has all the new RGN reps except for 29.
    """
    col_names = ['sequence', 'phenotype', 'is_train', 'is_test', 'phenotype_name',
       'dataset', 'RGN', '64_avg_hidden', '64_final_hidden', '64_final_cell',
       '256_avg_hidden', '256_final_cell', 'avg_hidden', 'final_hidden',
       'final_cell', 'arnold_original_3_7', 'arnold_scrambled_3_5',
       'arnold_random_3_7', 'arnold_uniform_4_1', 'all_64', 'all_256',
       'all_1900', 'all_avg_hidden', 'all_final_cell', 'RGN_avg_hidden',
       'RGN_final_cell']
    return pd.DataFrame(np.load(file), columns=col_names)


def get_tvt(path_to_dataset, dataset, phenotype_name, rep,
            modifiers=[tvt_modifier_baseline_reps,
                       tvt_modifier_return_mean,
                       tvt_modifier_scale_x
                       ],
            drop_strugglers=True, dataset_filetype='pkl', verbose=False,
            all_but_one_phenotype=False,
            is_transfer=False):

    """Returns training, validate, test dataframes with columns: rep, target

    Accepts a list of modifier functions
    Returned dataframe's 'rep' column is a list of numbers in each row; 'target'
    is a number (regression) or class name (classification) to predict

    Args:
        path_to_dataset (str): path to .npy dataset file
        dataset (str): name of dataset to filter by
        phenotype_name (str): name of phenotype to filter by
        rep_out (str): name of representation to filter by
        modifiers (list): a list of tvt_modifier functions that
            accept dataset, phenotype_name, rep, train, validate, test, verbose,
            all_but_one_phenotype, is_transfer
            and return train, validate, test
            modifiers can have side effects like saving necessary objects
        drop_strugglers (bool)
        dataset_filetype (str): "pkl", "npy", "dask", "loaded_full_dataset" - to be able to load
            various formats in which we have saved datasets before
        verbose (bool)
        all_but_one_phenotype (bool): if True, return train, validate, test
            split containing all phenotypes but the one specified in
            phenotype_name (useful for transfer); phenotype_name is required,
            and this only works if the dataset has multiple unique
            phenotype_names
        is_transfer (bool): are the reps being computed for transfer (as
            opposed to in-domain regression)
    """

    if verbose:
        print("Loading dataset file:", path_to_dataset)

    rep_out = rep[:] #  create a copy of rep

    if rep == 'mean':
        rep_out = "64_avg_hidden"
        #  just a small rep to avoid nan warnings - this will be replaced
        #  by mean of the target variable in
        #  the tvt_modifier_return_mean

    if is_baseline_rep(rep):
        rep_out = "sequence"
        #  need to pass sequence to calculate baseline reps later, in the
        #  modifier function tvt_modifier_baseline_reps

    #  Load dataset
    if dataset_filetype == 'pkl': #  Just use whatever is in pickle as the
        #  filtered dataset
        final = pd.read_pickle(path_to_dataset)

        if all_but_one_phenotype:
            if not is_transfer:
                raise ValueError(
                        f"Expected is_transfer==True with \
                        all_but_one_phenotype==True, but is_transfer is False"
                    )
            if 'full' not in path_to_dataset:
                raise ValueError(
                        f"Bad path_to_dataset: {path_to_dataset}, need the \
                        path to full dataset file to cut all_but_one_phenotype"
                    )
            if not len(subsets[dataset]) > 1:
                raise ValueError(
                        f"Invalid dataset: {path_to_dataset}, need dataset \
                        with multiple unique phenotype_name values to cut \
                        all_but_one_phenotype"
                    )
            final = final[(final.dataset == dataset) &
                  (final.phenotype_name != phenotype_name)]
            if verbose: print(f"Included all phenotypes but {phenotype_name}")

    elif dataset_filetype == 'loaded_full_dataset':
        final = path_to_dataset

        if all_but_one_phenotype:
            if not is_transfer:
                raise ValueError(
                        f"Expected is_transfer==True with \
                        all_but_one_phenotype==True, but is_transfer is False"
                    )
            if not len(subsets[dataset]) > 1:
                raise ValueError(
                        f"Invalid dataset: specified as {dataset}, need dataset \
                        with multiple unique phenotype_name values to cut \
                        all_but_one_phenotype"
                    )
            final = final[final.phenotype_name != phenotype_name]
            if verbose: print(f"Included all phenotypes but {phenotype_name}")
        elif phenotype_name == 'full':
            pass
        elif phenotype_name != 'full':
            final = final[final.phenotype_name == phenotype_name].copy()

    elif dataset_filetype == 'npy': #  This takes ~7minutes, cuts big npy
        #  I only include it for completeness, very much preferred not to use
        #  it
        final = npy_to_df(path_to_dataset)
        final = final[(final.dataset == dataset) &
                  (final.phenotype_name == phenotype_name)]

        if all_but_one_phenotype:
            #  TODO
            pass
    elif dataset_filetype == 'dask':
        #  TODO
        raise NotImplementedError("dask")
    else:
        raise ValueError(f"Invalid dataset_filetype: {dataset_filetype}")

    if rep_out == "RGN":
        final_df = final.loc[:, [rep_out, "sequence", "phenotype",
                                 "is_train", "is_test",
                                 "phenotype_name"]].copy()
    elif rep_out == "sequence":
        final_df = final.loc[:, [rep_out, "RGN", "phenotype",
                                 "is_train", "is_test",
                                 "phenotype_name"]].copy()
    elif rep_out == "phenotype_name": # for lopo splits
        final_df = final.loc[:, [rep_out, "RGN", "phenotype",
                                 "is_train", "is_test"]].copy()
    else:
        final_df = final.loc[:, [rep_out, "RGN", "sequence", "phenotype",
                                 "is_train", "is_test",
                                 "phenotype_name"]].copy()
    if verbose:
        print(f"filtered by representation {rep_out}")

    if drop_strugglers:
        if verbose:
            print(f"Dropping strugglers, initial shape: {final_df.shape}")
        final_df = final_df[np.isnan(
                    np.asarray(final_df.RGN.values.tolist())
                ).sum(axis=1) == 0]  #  this removes all rows where RGN rep is
        #  an array of NaNs (i.e. still computing)
        if verbose:
            print(f"Final Shape: {final_df.shape}")

    cols = ['rep', 'target']
    train = pd.DataFrame(columns=cols)
    validate = pd.DataFrame(columns=cols)
    test = pd.DataFrame(columns=cols)

    # Use the boolean columns in the dataframe to get train validate test split
    if dataset in auc_datasets.values:
        if verbose:
            print("dataset auc, skipping ttv split")
        train['rep']   =  final_df[final_df.is_train][rep_out].values.tolist()
        train['target']=  final_df[final_df.is_train]['phenotype'].values

        validate = train.copy()
        test = train.copy()
    else:
        train['rep']   =  final_df[(final_df.is_test == False) & (final_df.is_train)         ][rep_out].values.tolist()
        validate['rep'] = final_df[(final_df.is_test == False) & (final_df.is_train == False)][rep_out].values.tolist()
        test['rep']  =    final_df[(final_df.is_test)          & (final_df.is_train == False)][rep_out].values.tolist()

        train['target']=    final_df[(final_df.is_test == False) & (final_df.is_train)         ]['phenotype'].values
        validate['target'] =final_df[(final_df.is_test == False) & (final_df.is_train == False)]['phenotype'].values
        test['target'] =    final_df[(final_df.is_test)          & (final_df.is_train == False)]['phenotype'].values

    #  Apply modifiers
    for modifier in modifiers:
        train, validate, test = modifier(dataset, phenotype_name, rep,
                                               train, validate, test, verbose,
                                               all_but_one_phenotype,
                                               is_transfer)

    return train, validate, test

#  The below is for special purposes like returning mean
class NullModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X

def make_predictions(train, validate, test, metrics, model=None,
                     run_type='validate', verbose=True, to_train=True):
    """ Trains model, returns predictions series, trained model object, and
    results series (index=metrics, values=scores accroding to each metric)

    Args:
        train, validate, test (pd.DataFrame, pd.DataFrame, pd.DataFrame):
            dataframes with columns ['rep', 'target']
        metrics (dict of sklearn metrics classes): metrics to use in results
            evaluation (r^2, mse, accuracy etc)
        model (obj): sklearn-like model object with .fit() and .predict()
            methods
            if None, use a special class that just returns representation
            (for easy implementation of mean prediction and alike)
        run_type (str): 'validate' or 'test'
    """

    if model is None:
        model = NullModel()

    if to_train:
        model.fit(np.asarray(train['rep'].values.tolist()), train['target'].values.astype('float'))
        if verbose: print("Trained model")

    if run_type == 'test':
        validate = test

    predictions = model.predict(np.asarray(validate['rep'].values.tolist()))
    if verbose: print("Predicted")

    results = pd.Series()

    for metric_name in metrics.keys():
        #print(validate['target'].values)
        #print(predictions)
        results.loc[metric_name] = metrics[metric_name](validate['target'],
                   predictions)

    return predictions, model, results






# from code in ProDec-BLSTM https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1842-2#Sec15
def xroc(res, cutoff):
    """
    :type res: List[List[label, score]]
    :type curoff: all or 50
    """
    area, height, fp, tp = 0.0, 0.0, 0.0, 0.0
    for x in res:
        label = x
        if cutoff > fp:
            if label == 1:
                height += 1
                tp += 1
            else:
                area += height
                fp += 1
        else:
            if label == 1:
                tp += 1
    lroc = 0
    if fp != 0 and tp != 0:
        lroc = area / (fp * tp)
    elif fp == 0 and tp != 0:
        lroc = 1
    elif fp != 0 and tp == 0:
        lroc = 0
    return lroc


def get_roc(y_true, y_pred, cutoff):
    '''

    :param y_true:
    :param y_pred:
    :param cutoff:
    :return:
    '''
    score = []
    label = []

    for i in range(y_pred.shape[0]):
        label.append(y_true[i])
        score.append(y_pred[i])

    index = np.argsort(score)
    index = index[::-1]
    t_score = []
    t_label = []
    for i in index:
        t_score.append(score[i])
        t_label.append(label[i])

    score = xroc(t_label, cutoff)
    return score

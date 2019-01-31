"""
Usage:

from vectorize_for_benchmarking_lib import from_sequence_csv_to_X_and_dict, get_representations
import pandas as pd

path = "./path_to_training_sequences.csv"
path2 = "./path_to_test_sequences.csv"

X, dictionary, t = from_sequence_csv_to_X_and_dict(path)
X_tfidf, dictionary, t = from_sequence_csv_to_X_and_dict(path, ngramness=3, tfidf=True)
X_1000, dictionary, t = from_sequence_csv_to_X_and_dict(path, keep_n=1000) # limit bag of words to 1000 most frequent ngrams

#BUILDING REPRESENTATION FOR ANOTHER DATASET (I.E. TEST SET) AFTER "TRAINING":

#first, call it on the first dataset
X, dictionary_X, model_tfidf_X = from_sequence_csv_to_X_and_dict(path, tfidf=True)
#then pass that dictionary when you want representations for the second dataset
array_of_strings = pd.read_csv(path2).values[:,0]
Y, d, t = get_representations(array_of_strings, tfidf=True, dictionary=dictionary_X, model_tfidf = model_tfidf_X)

"""


import pandas as pd
import numpy as np
import nltk
# from gensim import corpora, similarities, models
from sklearn.linear_model import LassoLarsCV, LinearRegression, SGDRegressor, LassoLars, LassoCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, accuracy_score
#from gensim.matutils import corpus2dense
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def get_aa_percentage_vectors(X):
    res = np.zeros((X.shape[0], 20))
    for i,seq in enumerate(X):
        analysed_seq = ProteinAnalysis(seq)
        res[i] = pd.Series(analysed_seq.get_amino_acids_percent())[
            aas # to ensure the same order every time just in case
        ].values
    return res

def get_biopython_features(X):
    res = np.zeros((X.shape[0], 6))
    for i,seq in enumerate(X):
        analysed_seq = ProteinAnalysis(seq)
        res[i] = np.array([analysed_seq.molecular_weight()]+[analysed_seq.instability_index()] + [analysed_seq.isoelectric_point()] + list(analysed_seq.secondary_structure_fraction()))
        
    return res
    
def get_bow_representations(corpus, vectorizer = False, ngramness=3, max_features=10000):
    
    # corpus is an iterable of strings
    # returns an list of lists (each list a representation vector) 
    
    if not vectorizer:
        vectorizer = CountVectorizer(analyzer='char', 
                                     ngram_range=(ngramness,ngramness), 
                                     lowercase=False,
                                     max_features=max_features)
    
        X = vectorizer.fit_transform(corpus) 
        
    else:
        X = vectorizer.transform(corpus)
    
    return X.todense().tolist(), vectorizer

def transform_bow_to_tfidf(counts, transformer=False):
    if not transformer:
        transformer = TfidfTransformer()
        X = transformer.fit_transform(counts).toarray()
    else:
        X = transformer.transform(counts).toarray()
    
    return X.tolist(), transformer
    
# THESE FUNCTIONS ARE NOT GENERAL - USE WITH CARE - expect X to be a Series of string aminoacid sequences
    
def get_dumb_features(X):
    arr = np.concatenate((get_aa_percentage_vectors(X),  # frequencies 
                    get_biopython_features(X), # couple of simple biophysical features 
                    X.map(len).values.reshape(X.shape[0],1)), axis=1) # length of protein
    return arr.tolist()

def get_freq_and_len(X):
    arr = np.concatenate((get_aa_percentage_vectors(X),  # frequencies 
                    X.map(len).values.reshape(X.shape[0],1) / float(337) ), axis=1) # length of protein divided by avg length of prot in uniprot
    return arr.tolist()
    
def add_reps_to_mse(reps_names, mse, train, validate):
    # Accepts an array of representation names - must be columns in both train and validation dfs
    # where each row is a representation in array form.
    # 
    # 
    # Runs LassoLarsCV regerssion to predict relevant parameter, records training and validation mse,
    # returns an updated mse table.
    # Train and validation df must have representation columns and a "sol" column with target values
    
    lars_models = {
        rep:LassoLarsCV(
            n_jobs=-1
        ).fit(
            np.asarray(train[rep].values.tolist()), train['sol'].values
        ) for rep in reps_names
    }

    models = lars_models

    for rep in reps_names:
        X = np.asarray(validate[rep].values.tolist())
        yhat = models[rep].predict(X)
        mse.loc[rep, 'validation'] = mean_squared_error(validate['sol'], yhat)

        X = np.asarray(train[rep].values.tolist())
        yhat = models[rep].predict(X)
        mse.loc[rep, 'train'] = mean_squared_error(train['sol'], yhat)
        
    return mse
    
def run_data_efficiency(reps_names, mse, train, validate, 
                        timepoints = [10, 20, 30, 40, 50, 100, 200, 500],
                        replicates = 5,
                        verbose = False):
    # same as add_reps_to_mse, updated mse with validation loss for different-size datasets
    
    new_reps = reps_names
    
    for i in range(replicates):
        train = train.sample(frac=1).reset_index(drop=True)
        for rep in new_reps:
            if verbose:
                print("\n" + str(i) + "    " + rep)
            for cutoff in timepoints:
                if verbose:
                    print(str(cutoff) + " ", end='')
                lars_model = LassoLarsCV(
                    max_iter=2000
                ).fit(
                    np.asarray(train[rep].values.tolist())[:cutoff,:], train['sol'].values[:cutoff]
                )

                X = np.asarray(validate[rep].values.tolist())
                yhat = lars_model.predict(X)
                mse.loc[rep, cutoff] = mean_squared_error(validate['sol'], yhat)
        if i == 0:
            mse_old = mse.copy()
        else:
            mse.loc[new_reps] = (mse.loc[new_reps].values + mse_old.loc[new_reps].values) / 2
            mse_old = mse.copy()
            
    return mse
    
def compute_ngram_representations_and_return_new_dfs(train, validate, n = 2, 
                                                     max_features=10000, vectorizer=False, transformer=False):
    

    if vectorizer and transformer:
    
        train[f"{n}grams"], v = get_bow_representations(train.seq, vectorizer=vectorizer)
        train[f"tfidf_{n}grams"], t = transform_bow_to_tfidf( np.asarray(train[f"{n}grams"].values.tolist()),
                                                             transformer=transformer)

        validate[f"{n}grams"], v = get_bow_representations(validate.seq, vectorizer=vectorizer)
        validate[f"tfidf_{n}grams"], t = transform_bow_to_tfidf(np.asarray(validate[f"{n}grams"].values.tolist()),
                                                                transformer=transformer)

        return train, validate, vectorizer, transformer
    
    else:
        train[f"{n}grams"], vectorizer = get_bow_representations(train.seq, ngramness=n, 
                                                    max_features=max_features)
        train[f"tfidf_{n}grams"], transformer = transform_bow_to_tfidf(np.asarray(train[f"{n}grams"].values.tolist()))

        validate[f"{n}grams"], v = get_bow_representations(validate.seq, vectorizer=vectorizer)
        validate[f"tfidf_{n}grams"], t = transform_bow_to_tfidf(np.asarray(validate[f"{n}grams"].values.tolist()), 
                                                                transformer=transformer)
        
        return train, validate, vectorizer, transformer
        
def run_transfer(source_train, source_validate, target_train, target_validate, rep, baseline_rep = 'simple_freq_and_len'):
    
    
    source_train, source_validate, vectorizer, transformer  = compute_ngram_representations_and_return_new_dfs(
        source_train, source_validate)
    # We obtain a dictionary and a tf_idf model based exclusively on source domain data here - and then
    # use that to generate ngram and tfidf representations for the target domain
    
    #print(source_train.columns)
    #print(source_train.iloc[:5,-1])
    
    target_train, target_validate, bla, blabla = compute_ngram_representations_and_return_new_dfs(
        target_train, target_validate,  vectorizer=vectorizer, transformer=transformer)

    source_train = source_train.sample(frac=1).reset_index(drop=True)
    target_train = target_train.sample(frac=1).reset_index(drop=True)
    target_validate = target_validate.sample(frac=1).reset_index(drop=True)
        
    model_trained_on_S = LassoLarsCV(
            n_jobs=-1, max_iter=500
        ).fit(np.asarray(source_train[rep].values.tolist()), source_train['sol'].values)
    
    model_trained_on_T = LassoLarsCV(
            n_jobs=-1, max_iter=500
        ).fit(np.asarray(target_train[rep].values.tolist()), target_train['sol'].values)
    
    baseline_trained_on_T = LassoLarsCV(
            n_jobs=-1, max_iter=500
        ).fit(np.asarray(target_train[baseline_rep].values.tolist()), target_train['sol'].values)
    
    errors = pd.Series()
        
    # Transfer Error
    X = np.asarray(target_validate[rep].values.tolist())
    yhat = model_trained_on_S.predict(X)
    errors.loc['error_S_T'] = mean_squared_error(target_validate['sol'], yhat) 

    #In-domain Error
    X = np.asarray(target_validate[rep].values.tolist())
    yhat = model_trained_on_T.predict(X)
    errors.loc['error_T_T'] = mean_squared_error(target_validate['sol'], yhat)
    
    # Baseline in-domain error
    X = np.asarray(target_validate[baseline_rep].values.tolist())
    yhat = baseline_trained_on_T.predict(X)
    errors.loc['b_error_T_T'] = mean_squared_error(target_validate['sol'], yhat)
        
    
    errors.loc['transfer_ratio']  = errors.loc['error_S_T'] / float(errors.loc['b_error_T_T'])
    errors.loc['indomain_ratio'] = errors.loc['error_T_T'] / float(errors.loc['b_error_T_T'])
    
    return errors

def run_all_transfers(reps_names, results1v1, results_train_one_eval_all, results_eval_on_holdout, domain_combos, dfs_by_organsim, domain_names):
    
    ## if first_run=False, needs to have loaded 

    print("!1 V 1!")
    results1v1.columns = ['transfer_ratio_avg', 'indomain_ratio_avg']

    for rep in reps_names:

        print(rep)

        errors = pd.DataFrame(index=domain_combos, columns=['error_S_T', 'error_T_T', 'b_error_T_T', 'transfer_ratio',
               'indomain_ratio'])

        for source_name,target_name in errors.index:

            er = run_transfer(source_train= dfs_by_organsim[source_name]['train'], 
                     source_validate= dfs_by_organsim[source_name]['validate'],
                     target_train= dfs_by_organsim[target_name]['train'], 
                     target_validate= dfs_by_organsim[target_name]['validate'], 
                     rep=rep
            )
            errors.loc[(source_name,target_name),:] = er.values
            print(f"    {source_name}->{target_name} complete")

        results1v1.loc[rep,:] = errors[['transfer_ratio', 'indomain_ratio']].mean().values

    print("!train_one_eval_all!")
    results_train_one_eval_all.columns = ['transfer_ratio_avg', 'indomain_ratio_avg']
    
    for rep in reps_names:
        print(rep)
        for name in domain_names:
            print(f"    {name}, evaluated on all others")
            others = np.setdiff1d(domain_names,[name])

            errors = pd.DataFrame(index=domain_names, columns=['error_S_T', 'error_T_T', 'b_error_T_T', 'transfer_ratio',
                   'indomain_ratio'])

            er = run_transfer(source_train= dfs_by_organsim[name]['train'], 
                         source_validate= dfs_by_organsim[name]['validate'],

                         target_train=pd.concat([dfs_by_organsim[other_name]['train'] for other_name in others]), 

                         target_validate= pd.concat([dfs_by_organsim[other_name]['validate'] for other_name in others]), 
                         rep=rep
                )
            errors.loc[name,:] = er.values

        results_train_one_eval_all.loc[rep,:] = errors[['transfer_ratio', 'indomain_ratio']].mean().values

    print("!eval_on_holdout!")
    results_eval_on_holdout.columns = ['transfer_ratio_avg', 'indomain_ratio_avg']
    
    for rep in reps_names:
        print(rep)
        for name in domain_names:
            print(f"    trained on all but, evaluated on {name}")
            others = np.setdiff1d(domain_names,[name])

            errors = pd.DataFrame(index=domain_names, columns=['error_S_T', 'error_T_T', 'b_error_T_T', 'transfer_ratio',
                   'indomain_ratio'])

            er = run_transfer(
                source_train    = pd.concat([dfs_by_organsim[other_name]['train'] for other_name in others]), 
                source_validate = pd.concat([dfs_by_organsim[other_name]['validate'] for other_name in others]),

                target_train= dfs_by_organsim[name]['train'], 

                target_validate= dfs_by_organsim[name]['validate'], 
                rep=rep
            )
            errors.loc[name,:] = er.values

        results_eval_on_holdout.loc[rep,:] = errors[['transfer_ratio', 'indomain_ratio']].mean().values
    
    return results1v1, results_train_one_eval_all, results_eval_on_holdout

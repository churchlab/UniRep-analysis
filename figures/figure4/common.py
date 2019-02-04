import sys
import os
import subprocess

name_to_dataset = {
    'gfp_sarkisyan_unirep': 's3://protein-representation-learning/mlpe_gfp/gfp_avg_hidden_sark_bf_fixed_rep_tts.npz',
    'gfp_sarkisyan_evotuned_unirep': 's3://protein-representation-learning/mlpe_gfp/evotune/sark_sfGFP_finetune1_avg_hidden.npz',
    'gfp_gen_set_evotuned_unirep': (
        's3://protein-representation-learning/mlpe_gfp/generalization_mechanism/finetune1_fpbase_reps.npy'),
    'gfp_sarkisyan_evotuned_random': 's3://protein-representation-learning/mlpe_gfp/evotune/gfp_1random_13120__avg_hidden.npz',
    'gfp_sarkisyan_baseline_reps': (
        's3://protein-representation-learning/mlpe_gfp/generalization_mechanism/reps_sarkisyan.dict.joblibpkl'),
    'gfp_gen_set_baseline_reps': (
        's3://protein-representation-learning/mlpe_gfp/generalization_mechanism/all_fpbase_reps.dict.joblibpkl')
}

def dataset_name_to_local_path(dataset_name, local_dir):
    return os.path.join(local_dir, name_to_dataset[dataset_name].split('/')[-1])

def sync_dataset(dataset_name, destination):
    cmd = ('aws s3 cp --no-sign-request ' + name_to_dataset[dataset_name] + 
           ' ' + destination)
    print(cmd)
    subprocess.check_output(cmd, shell=True)
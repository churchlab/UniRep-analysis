# UniRep-analysis
Analysis and figure code from Alley et al. 2019.

Start by cloning the repo:
```
git clone https://github.com/churchlab/UniRep-analysis.git
```

# Requirements
python: 3.5.2

For reference on how to install, see https://askubuntu.com/questions/682869/how-do-i-install-a-different-python-version-using-apt-get

venv with necessary requirements, can be installed with:
```
cd UniRep-analysis # root directory of the repository
python3 -m venv venv/
source venv/bin/activate
pip install -r venv_requirements/requirements-py3.txt
deactivate
```

conda, can be installed with:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# restart the currrent shell
```

two conda environments necessary to run different parts of the code:
```
cd UniRep-analysis # root directory of the repository
conda env create -f ./yml/grig_alldatasets_run.yml
conda env create -f ./yml/ethan_analysis.yml
```

# Getting the data
```
mkdir data
cd data

wget https://s3.us-east-2.amazonaws.com/unirep-data-storage/unirep_analysis_data_part2.tar.gz
tar -zxvf unirep_analysis_data_part2.tar.gz # this may take some time
rm unirep_analysis_data_part2.tar.gz

wget https://s3.amazonaws.com/unirep-public/unirep_analysis_data.zip
unzip unirep_analysis_data.zip
rm unirep_analysis_data.zip

cd ..
```

# Project Structure

# Usage
## Reproducing figures

To re-generate all the figures in the main text, one should execute jupyter/ipython notebooks in the /figures directory using the right environment.

To run a notebook, do the following:

Activate the right environment (as noted in the Project Structure section for each notebook):

For grig_alldatasets_run:
```
source activate grig_alldatasets_run
```
For ethan_analysis:
```
source activate ethan_analysis
```
For venv:
```
source venv/bin/activate
```

Then execute:
```
jupyter notebook
```
This will automatically open a browser window where one can interactively rerun the code generating the figures. Aesthetic components (colors, font sizes) may differ slightly from the final version of the figures in the paper.

## Re-training top models and re-generating performance metrics

In order to re-train the models and regenerate the metrics from which the figures are constructed, one can run the python scripts in the /analysis folder. By default these will evaluate all representations and baselines on all available datasets and subsets and will computer metrics (such as MSE and Pearson r) on the test subset.

The easiest way is to do this is to start an AWS instance with sufficient resources (we recommend m5.12xlarge or m5.24xlarge for shorter runtime - the code takes advantage of all the available CPU cores) with Ubuntu Server 18.04 LTS AMI (for example, ami-0f65671a86f061fcd). After performing the initial setup above, create the necessary directories:
```
cd analysis
mkdir results # folder for various model performance metrics
mkdir predictions # folder for model predictions for various datasets
mkdir models # folder for trained models
mkdir params # folder for recording best parameters after hyperparameter search 
```

To run SCOP 1.67 Superfamily Remote Homology Detection and SCOP 1.67 Fold-level Similarity Detection with Random Forest, execute:
```
python FINAL_run_RF_homology_detection.py
```

To run quantitative function prediction, de novo designed mini proteins stability prediction, DMS stability prediction for 17 de novo designed and natural protein datasets from Figure 3, as well as supplementary benchmarks, such as small-scale function prediction (Supp. Table S4):
```
python FINAL_run_l1_regr_quant_function_stability_and_supp_analyses.py
python FINAL_compute_std_by_val_resampling.py # computes estimates of standard deviations through validation/test set resampling for significance testing (generates std_results_val_resamp.csv)
```

To run the analyses from Supp. Fig. S10 (generalized stability prediction, generalized quantitative function prediction and a special central to remote generalized stability prediction task):
```
python FINAL_run_transfer_analysis_function_prediction__stability.py
```

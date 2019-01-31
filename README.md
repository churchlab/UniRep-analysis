# UniRep-analysis
Analysis and figure code from Alley et al. 2019.

There is a really simple set-up necessary to reproduce the analysis.
One needs a Linux (tested on Ubuntu 16.04) machine with a web browser (for running jupyter notebooks).

Step 1: Downdload data & install conda (a python package manager that will install all necessary packages for the analysis): download installer and follow the interactive process

```
mkdir data
cd data
wget https://s3.us-east-2.amazonaws.com/unirep-data-storage/unirep_analysis_data_part2.tar.gz
tar -zxvf unirep_analysis_data_part2.tar.gz # this may take some time
rm unirep_analysis_data_part2.tar.gz
cd ..

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
```
At this step one may need to restart the gterminal to be able to use the newly installed conda



Step 2: Clone the github repo with analysis code and create a conda environment including all the necessary packages to execute analysis, add the second part of the data
```
git clone https://github.com/churchlab/UniRep-analysis.git
cd UniRep-analysis
conda env create -f ./yml/grig_alldatasets_run.yml
source activate grig_alldatasets_run
mkdir data
cd data
wget https://s3.amazonaws.com/unirep-public/unirep_analysis_data.zip
unzip unirep_analysis_data.zip
rm unirep_analysis_data.zip
cd ..
```
______________________________________________________

To re-generate all the figures in the main text, one should execute jupyter/ipython notebooks in the /figures directory.
Executing:
```
jupyter notebook
```
after the set-up described above will automatically open a browser window where one can interactively rerun the code generating the figures. Aesthetic components (colors, font sizes) may differ slightly from the final version of the figures in the paper.

______________________________________________________

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

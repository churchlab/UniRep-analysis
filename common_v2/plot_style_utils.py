"""
Utilities for plotting beautiful figures.
"""
import matplotlib.pyplot as plt
import palettable as pal
import seaborn as sns
import pandas as pd

def prettify_ax(ax):
    """
    Nifty function we can use to make our axes more pleasant to look at
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frameon = True
    ax.patch.set_facecolor('#eeeeef')
    ax.grid('on', color='w', linestyle='-', linewidth=1)
    ax.tick_params(direction='out')
    ax.set_axisbelow(True)


def simple_ax(figsize=(6, 4), **kwargs):
    """
    Shortcut to make and 'prettify' a simple figure with 1 axis
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **kwargs)
    prettify_ax(ax)
    return fig, ax

def set_pub_plot_context(colors='categorical', context="talk"):
    if colors == 'categorical':
        palette = pal.cartocolors.qualitative.Safe_10.mpl_colors

    sns.set(palette=palette, style="whitegrid", context=context, font="Arial")
    return palette


def save_for_pub(fig, path="../../data/default", dpi=1000):
    fig.savefig(path + ".png", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".eps", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".svg", dpi=dpi, bbox_inches='tight')
    #fig.savefig(path + ".emf", dpi=dpi, bbox_inches='tight')
   #fig.savefig(path + ".tif", dpi=dpi)

rep_names ={
    'RGN': 'RGN',
    '64_avg_hidden': 'UniRep 64-unit Avg. Hidden State',
    '64_final_hidden': 'UniRep 64-unit Final Hidden State',
    '64_final_cell': 'UniRep 64-unit Final Cell State',
    '256_avg_hidden': 'UniRep 256-unit Avg. Hidden State',
    '256_final_cell': 'UniRep 256-unit Final Hidden State',
    'avg_hidden': 'UniRep 1900-unit Avg. Hidden State',
    'final_hidden': 'UniRep 1900-unit Final Hidden State',
    'final_cell': 'UniRep 1900-unit Final Cell State',
    'arnold_original_3_7': 'Doc2Vec Original k=3 w=7',
    'arnold_scrambled_3_5': 'Doc2Vec Scrambled k=3 w=5',
    'arnold_random_3_7': 'Doc2Vec Random k=3 w=7',
    'arnold_uniform_4_1': 'Doc2Vec Uniform k=4 w=1',
    'all_64': 'UniRep 64-unit Fusion',
    'all_256': 'UniRep 256-unit Fusion',
    'all_1900': 'UniRep Fusion',
    'all_avg_hidden': 'UniRep 64-unit + 256-unit + 1900-unit Avg. Hiddens',
    'all_final_cell': 'UniRep 64-unit + 256-unit + 1900-unit Final Cells',
    'RGN_avg_hidden': 'RGN + UniRep 1900-unit Avg. Hiddens',
    'RGN_final_cell':'RGN + UniRep 1900-unit Final Cells',
    'simple_freq_plus': "Our Baseline: Amino Acid Freq. and Predicted Biophys. Params.",
    'simple_freq_and_len': "Our Baseline: Amino Acid Freq. and Protein Length",
    '2grams': "Our Baseline: 2-grams" ,
    '3grams': "Our Baseline: 3-grams",
    'tfidf_2grams': "Our Baseline: 2-grams with TF-IDF weighting",
    'tfidf_3grams': "Our Baseline: 3-grams with TF-IDF weighting",
    'sequence': "Levenshtein (generalized edit) distance",
    'mean': "Our Baseline: Dataset Mean"
}

# For naming reps in the main text
main_text_rep_names ={
    'RGN': 'RGN',
    'avg_hidden': 'UniRep',
    'arnold_original_3_7': 'Doc2Vec Original k=3 w=7',
    'arnold_scrambled_3_5': 'Doc2Vec Scrambled k=3 w=5',
    'arnold_random_3_7': 'Doc2Vec Random k=3 w=7',
    'arnold_uniform_4_1': 'Doc2Vec Uniform k=4 w=1',
    'all_1900': 'UniRep Fusion',
    'all_256': 'UniRep 256-unit Fusion',
    'RGN_avg_hidden': 'RGN-UniRep Fusion',
    'RGN_final_cell':'RGN-UniRep Cell Fusion',
    'simple_freq_plus': "AA. Freq. + Predicted Biophys. Params.",
    'simple_freq_and_len': "AA. Freq.+ Length",
    '2grams': "2-grams" ,
    '3grams': "3-grams",
    'tfidf_2grams': "TFIDF 2-grams",
    'tfidf_3grams': "TFIDF 3-grams",
    'sequence': "Levenshtein",
    'best_other_rep':'Our Best Baseline',
    'best_arnold_rep': 'Best Doc2Vec'
}

task_names = {

    'arnold_T50': "Cytochrome P450 Thermostability",
    'arnold_absorption': 'Rhodopsin Peak Absorption Wavelength',
    'arnold_enantioselectivity': 'Epoxide Hydrolase Enantioselectivity',
    'arnold_localization': 'Channelrhodopsin Membrane Localization',

    'fowler': 'Variant Effect Prediction Task',
    'fowler_consistent_single_UBI': 'Variant Effect Prediction Task',

    'TEM-1_variant_score':"TEM-1 Beta-lactamase",
    'E1_Ubiquitin':"Ubiquitin (E1 Activity)",
    'gb1_variant_score':"Protein G (IgG domain)",
    'hsp90_variant_score':"HSP90",
    'Kka2_variant_score':"Aminoglycosidase (Kka2)",
    'Pab1_variant_score':"Pab1 (RRM domain)",
    'PSD95pdz3_variant_score':"PSD95 (Pdz3 domain)",
    'Ubiquitin':"Ubiquitin",
    'Yap65_variant_score':"Yap65 (WW domain)",

    'natural':'Natural',
    'engineered':'De-Novo Designed',

    'full':'Combined Data',

    'rocklin_ssm2': "Natural & De-Novo Designed Proteins Stability - Site Saturation Mutagenesis",
    'rocklin_ssm2_nat_eng': "Natural & De-Novo Designed Mutant Proteins Stability - Site Saturation Mutagenesis",
    'rocklin_all_rds': "De-Novo Designed Proteins Stability - Design Rounds",
    'rocklin_ssm2_remote_test': "Natural & De-Novo Designed Mutant Proteins Stability - Remote",

    'homstrad': 'Protein Family Prediction from HOMSTRAD Database', #; Mizuguchi 1998',
    'leuenberger': 'Thermostability Prediction Across Organisms', #; Leuenberger 2017',
    'oxbench': 'OXBENCH Database Reference Alignment Clustering', #; Raghava 2003",

    'solubility':'Protein Solubility Prediction', #; Uemura 2018, Niwa 2009',
    'handstad_folds': 'SCOP 1.67 Superfamily Remote Homology Detection Benchmark', #; Håndstad 2007',
    'handstad_superfamily': 'SCOP 1.67 Folds Remote Homology Detection Benchmark', #; Håndstad 2007',

    'EEHEE_rd3_0037.pdb_ssm2_stability': 'EEHEE_rd3_0037',
    'EEHEE_rd3_1498.pdb_ssm2_stability': 'EEHEE_rd3_1498',
    'EEHEE_rd3_1702.pdb_ssm2_stability': 'EEHEE_rd3_1702',
    'EEHEE_rd3_1716.pdb_ssm2_stability': 'EEHEE_rd3_1716',
    'EHEE_0882.pdb_ssm2_stability': 'EHEE_0882',
    'EHEE_rd2_0005.pdb_ssm2_stability': 'EHEE_rd2_0005',
    'EHEE_rd3_0015.pdb_ssm2_stability': 'EHEE_rd3_0015',
    'HEEH_rd2_0779.pdb_ssm2_stability': 'HEEH_rd2_0779',
    'HEEH_rd3_0223.pdb_ssm2_stability': 'HEEH_rd3_0223',
    'HEEH_rd3_0726.pdb_ssm2_stability': 'HEEH_rd3_0726',
    'HEEH_rd3_0872.pdb_ssm2_stability': 'HEEH_rd3_0872',
    'HHH_0142.pdb_ssm2_stability': 'HHH_0142',
    'HHH_rd2_0134.pdb_ssm2_stability': 'HHH_rd2_0134',
    'HHH_rd3_0138.pdb_ssm2_stability': 'HHH_rd3_0138',
    'Pin1_ssm2_stability': 'Pin1',
    'hYAP65_ssm2_stability': 'hYAP65',
    'villin_ssm2_stability': 'villin',

    'ecoli_solubility_score':'E. Coli',
    'yeast_solubility_score':'S. Cerevisiae',
    'human_tm':'H. Sapiens',
    'ecoli_tm':'E. Coli',
    'thermophilus_tm':'S. Thermophilus',
    'yeast_tm':'S. Cerevisiae'


}




def label_point(x, y, val, ax, fontsize=20):
    """
    Label x, y points on a given ax with val text of a particular
    fontsize
    """
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']),
               fontsize=fontsize)


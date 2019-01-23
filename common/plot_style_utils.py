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
	if colors == "sequential":
		palette = pal.colorbrewer.sequential.YlOrBr_9.mpl_colors
	sns.set(
		palette=palette,
		style="whitegrid",
		context=context,
            font="Arial"
	)



def save_for_pub(fig, path="../../data/default", dpi=1000):
    fig.savefig(path + ".png", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".eps", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(path + ".svg", dpi=dpi, bbox_inches='tight')
    # fig.savefig(path + ".tif", dpi=dpi)

""" 
rep_names ={
    'RGN': 'RGN, AlQuraishi 2018', 
    '64_avg_hidden': 'UniRep 64-unit Avg. Hidden State', 
    '64_final_hidden': 'UniRep 64-unit Final Hidden State', 
    '64_final_cell': 'UniRep 64-unit Final Cell State',
    '256_avg_hidden': 'UniRep 256-unit Avg. Hidden State', 
    '256_final_cell': 'UniRep 64-unit Final Hidden State', 
    'avg_hidden': 'UniRep 1900-unit Avg. Hidden State', 
    'final_hidden': 'UniRep 1900-unit Final Hidden State',
    'final_cell': 'UniRep 1900-unit Final Cell State', 
    'arnold_original_3_7': 'Doc2Vec Original 3_7, Yang 2018', 
    'arnold_scrambled_3_5': 'Doc2Vec Scrambled 3_5, Yang 2018',
    'arnold_random_3_7': 'Doc2Vec Random 3_7, Yang 2018', 
    'arnold_uniform_4_1': 'Doc2Vec Uniform 4_1, Yang 2018', 
    'all_64': 'UniRep 64-unit Concatenation of All States',
    'all_256': 'UniRep 256-unit Concatenation of All States',
    'all_1900': 'UniRep 1900-unit Concatenation of All States', 
    'all_avg_hidden': 'UniRep 64-unit + 256-unit + 1900-unit Avg. Hiddens', 
    'all_final_cell': 'UniRep 64-unit + 256-unit + 1900-unit Final Cells', 
    'RGN_avg_hidden': 'RGN, AlQuraishi 2018 + UniRep 1900-unit Avg. Hiddens',
    'RGN_final_cell':'RGN, AlQuraishi 2018 + UniRep 1900-unit Final Cell', 
    'simple_freq_plus': "Baseline: Amino Acid Freq. and Predicted Biophys. Params.", 
    'simple_freq_and_len': "Baseline: Amino Acid Freq. and Protein Length", 
    '2grams': "Baseline: 2-grams" ,
    '3grams': "Baseline: 3-grams", 
    'tfidf_2grams': "Baseline: 2-grams with TF-IDF weighting", 
    'tfidf_3grams': "Baseline: 3-grams with TF-IDF weighting",
    'sequence': "Levenshtein (generalized edit) distance",
}
"""

rep_names ={
    'RGN': 'RGN', 
    '64_avg_hidden': 'UniRep 64-unit Avg. Hidden State', 
    '64_final_hidden': 'UniRep 64-unit Final Hidden State', 
    '64_final_cell': 'UniRep 64-unit Final Cell State',
    '256_avg_hidden': 'UniRep 256-unit Avg. Hidden State', 
    '256_final_cell': 'UniRep 64-unit Final Hidden State', 
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

main_text_rep_names ={
    'RGN': 'RGN', 
    'avg_hidden': 'UniRep', 
    'arnold_original_3_7': 'Doc2Vec Original k=3 w=7', 
    'arnold_scrambled_3_5': 'Doc2Vec Scrambled k=3 w=5',
    'arnold_random_3_7': 'Doc2Vec Random k=3 w=7', 
    'arnold_uniform_4_1': 'Doc2Vec Uniform k=4 w=1', 
    'all_1900': 'UniRep Fusion', 
    'RGN_avg_hidden': 'RGN-UniRep', 
    'simple_freq_plus': "AA. Freq. + Predicted Biophys. Params.", 
    'simple_freq_and_len': "AA. Freq.+ Length", 
    '2grams': "2-grams" ,
    '3grams': "3-grams", 
    'tfidf_2grams': "TFIDF 2-grams", 
    'tfidf_3grams': "TFIDF 3-grams",
    'sequence': "Levenshtein",
    'best_other_rep':'Our Best Baseline',
    'best_arnold_rep':'Best Doc2Vec'
}


def label_point(x, y, val, ax, fontsize=20, rotation=0):
	"""
	Label x, y points on a given ax with val text of a particular 
	fontsize
	"""
	a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
	for i, point in a.iterrows():
		ax.text(point['x']+.02, point['y'], str(point['val']),
               fontsize=fontsize, rotation=rotation)
       	

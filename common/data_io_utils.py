""" 
Functions for loading datasets and writing results
"""
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd


# Lookup tables
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
}

int_to_aa = {value:key for key, value in aa_to_int.items()}

def load_solubility_df(filepath):
    """
    Filename should be .npy file from google drive.
    Outputs correctly formated df.
    """
    names = ["seq", "sol", "final_cell", "final_hidden", "avg_hidden"]
    from_disk = pd.DataFrame(
        np.load(filepath),
        columns=names)
    from_disk['sol'] = from_disk['sol'].astype(float)
    return from_disk


def load_rocklin_df(filepath):
    """
    Filename should be .npy file from google drive.
    Outputs correctly formated df.
    """
    names = ["seq", "stability", "final_cell", "final_hidden", "avg_hidden"]
    from_disk = pd.DataFrame(
        np.load(filepath),
        columns=names)
    from_disk['stability'] = from_disk['stability'].astype(float)
    return from_disk


def load_leuenberger_df(filepath):
    """
    Filename should be .npy file from google drive.
    Outputs correctly formated df.
    """
    names = ["seq", "tm", "final_cell", "final_hidden", "avg_hidden"]
    from_disk = pd.DataFrame(
        np.load(filepath),
        columns=names)
    from_disk['tm'] = from_disk['tm'].astype(float)
    return from_disk


def load_fowler_df(filepath):
    """
    Filename should be .npy file from google drive.
    Outputs correctly formated df.
    """
    names = ["seq", "score", "id", "final_cell", "final_hidden", "avg_hidden"]
    from_disk = pd.DataFrame(
        np.load(filepath),
        columns=names)
    from_disk['score'] = from_disk['score'].astype(float)
    return from_disk






import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from scipy.sparse import diags
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

label_dict = {
    '.': np.array([1, 0, 0]),
    '(': np.array([0, 1, 0]),
    ')': np.array([0, 0, 1])
}
seq_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),  # T or U
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),

    'R': np.array([1, 0, 0, 1]),
    'Y': np.array([0, 1, 1, 0]),
    'K': np.array([0, 1, 0, 1]),
    'M': np.array([1, 0, 1, 0]),
    'S': np.array([0, 0, 1, 1]),
    'W': np.array([1, 1, 0, 0]),
    'B': np.array([0, 1, 1, 1]),
    'D': np.array([1, 1, 0, 1]),
    'H': np.array([1, 1, 1, 0]),
    'V': np.array([1, 0, 1, 1]),
    'N': np.array([0, 0, 0, 0]),
    '_': np.array([0, 0, 0, 0]),
    '~': np.array([0, 0, 0, 0]),
    '.': np.array([0, 0, 0, 0]),
    'P': np.array([0, 0, 0, 0]),
    'I': np.array([0, 0, 0, 0]),
    'X': np.array([0, 0, 0, 0])
}

char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('N')   # replace '.' to 'N'
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def contact_map_masks(data_lens, matrix_rep):
    n_seq = len(data_lens)
    assert matrix_rep.shape[0] == n_seq
    for i in range(n_seq):
        l = int(data_lens[i].cpu().numpy())
        matrix_rep[i, :l, :l] = 1
    return matrix_rep

# return index of contact pairing, index start from 0
def get_pairings(data):
    rnadata1 = list(data.loc[:, 0].values)
    rnadata2 = list(data.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    rna_pairs = (np.array(rna_pairs) - 1).tolist()
    return rna_pairs

# generate .dbn format
def generate_label_dot_bracket(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] <= 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return ''.join(rnastructure)


# extract the pseudoknot index given the data
def extract_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                print(i, j)
                break


def find_pseudoknot(data):
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    flag = False
    for i in range(len(rnadata2)):
        for j in range(len(rnadata2)):
            if (rnadata1[i] < rnadata1[j] < rnadata2[i] < rnadata2[j]):
                flag = True
                break
    return flag


def seq_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: seq_dict[x.upper()], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def struct_encoding(string):
    str_list = list(string)
    encoding = list(map(lambda x: label_dict[x], str_list))
    # need to stack
    return np.stack(encoding, axis=0)


def padding(data_array, maxlen):
    a, b = data_array.shape
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def plot__fig(name: str, data: list, sample_number: int, shortest_seq: int, longest_seq: int):
    plot_data = np.array(data)
    plt.figure(figsize=(16, 10))
    n, bins, patches = plt.hist(plot_data, bins=10, rwidth=0.8, align='left', color='b', edgecolor='white')
    for i in range(len(n)):
        plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    plt.title(
        f'{name} data distribution(Total number:{sample_number},shortest_seq:{shortest_seq},longest_seq:{longest_seq})'
    )
    plt.grid()
    # plt.legend()
    fig_name = f'{name}' + '.png'
    if not os.path.exists(fig_name):
        plt.savefig(fig_name)
    # plt.show()


def draw_dis(name: str, data: list, sample_number: int, shortest_seq: int, longest_seq: int):
    # draw the squence length distribution
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.histplot(data, kde=False, color='b')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Sequence length', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title(
        f'{name} data distribution(Total number:{sample_number},shortest_seq:{shortest_seq},longest_seq:{longest_seq})'
    )
    plt.savefig(f'{name}.png', dpi=200, bbox_inches='tight')

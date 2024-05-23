# -*- coding: utf-8 -*-
import os
from os.path import join
import munch
import random
import numpy as np
import pandas as pd
import torch
import math
from typing import List, Sequence, Tuple
import collections
from itertools import product
from collections import defaultdict
from models.model import DiffusionRNA2dPrediction
import json

seq_to_onehot_dict = {
    'A': np.array([1, 0, 0, 0]),
    'U': np.array([0, 1, 0, 0]),  # T or U
    'C': np.array([0, 0, 1, 0]),
    'G': np.array([0, 0, 0, 1]),
    'N': np.array([0, 0, 0, 0]),

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


PARENTHESES = [
    ("(", ")"),
    ("[", "]"),
    ("<", ">"),
    ("{", "}")
]


def process_config(jsonfile):
    with open(jsonfile, 'r') as f:
        config_dict = json.load(f)
    config = munch.Munch(config_dict)
    config.model = munch.Munch(config.model)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fasta(fasta_file_path):
    sequences = dict()
    current_id = None
    current_seq = []
    with open(fasta_file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line.strip()[1:]
                current_seq = []
            else:
                current_seq.append(line.strip().upper())

        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    return sequences


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('N')   # replace '.' to 'N'
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def seq2encoding(seq):
    encoding = list()
    for char in seq:
        encoding.append(seq_to_onehot_dict[char])
    return np.array(encoding)


def contact2ct(contact, seq, seq_len):
    contact = contact[:seq_len, :seq_len]
    structure = np.where(contact)
    pair_dict = dict()
    for i in range(seq_len):
        pair_dict[i] = -1
    for i in range(len(structure[0])):
        pair_dict[structure[0][i]] = structure[1][i]
    first_col = list(range(1, seq_len + 1))
    second_col = list(seq)
    third_col = list(range(seq_len))
    fourth_col = list(range(2, seq_len + 2))
    fifth_col = [pair_dict[i] + 1 for i in range(seq_len)]
    last_col = list(range(1, seq_len + 1))
    df = pd.DataFrame()
    df['index'] = first_col
    df['base'] = second_col
    df['index-1'] = third_col
    df['index+1'] = fourth_col
    df['pair_index'] = fifth_col
    df['n'] = last_col
    return df


def extract_pseudoknot(pairs):
    pseudo_pairs = list()
    for (i1, j1) in pairs:
        for (i2, j2) in pairs:
            if i1 < i2 < j1 < j2:
                pseudo_pairs.append((i2, j2))
    return pseudo_pairs


def contact2dbn(contact, seq_len):
    contact = contact[:,:seq_len, :seq_len]
    structure = np.where(contact)[1:]
    pairs = list(map(lambda i: (structure[0][i], structure[1][i]), range(len(structure[0]))))
    pairs_0 = [(i, j) for (i, j) in pairs if (j,i) in set(pairs) and i < j]
    pairs_dict = defaultdict(list)
    pk_pairs_1 = extract_pseudoknot(pairs_0)
    pk_pairs_2 = extract_pseudoknot(pk_pairs_1)
    pk_pairs_3 = extract_pseudoknot(pk_pairs_2)
    for index, pairs in enumerate([pairs_0, pk_pairs_1, pk_pairs_2, pk_pairs_3]):
        if len(pairs) != 0:
            pairs_dict[index] = pairs

    dbn = np.array(['.'] * seq_len)
    for index, pairs in pairs_dict.items():
        for (i, j) in pairs:
            dbn[i] = PARENTHESES[index][0]
            dbn[j] = PARENTHESES[index][1]
    dbn = ''.join(dbn)
    return dbn


def ct2dbn(ctfile):
    seq = ''.join(list(ctfile.loc[:, 1])).upper()
    seq_len = len(seq)
    rnadata1 = list(ctfile.loc[:, 0].values)
    rnadata2 = list(ctfile.loc[:, 4].values)
    rna_pairs = list(zip(rnadata1, rnadata2))
    rna_pairs = list(filter(lambda x: x[1] > 0, rna_pairs))
    pairs_0 = (np.array(rna_pairs) - 1).tolist()

    pairs_dict = defaultdict(list)
    pk_pairs_1 = extract_pseudoknot(pairs_0)
    pk_pairs_2 = extract_pseudoknot(pk_pairs_1)
    pk_pairs_3 = extract_pseudoknot(pk_pairs_2)
    for index, pairs in enumerate([pairs_0, pk_pairs_1, pk_pairs_2, pk_pairs_3]):
        if len(pairs) != 0:
            pairs_dict[index] = pairs

    dbn = np.array(['.'] * seq_len)
    for index, pairs in pairs_dict.items():
        for [i, j] in pairs:
            if i < j and [j, i] in pairs:
                dbn[i] = PARENTHESES[index][0]
                dbn[j] = PARENTHESES[index][1]
    dbn = ''.join(dbn)
    return (seq, dbn)


def get_data(file_path, alphabet):
    data_dict = parse_fasta(file_path)
    name_list = list()
    seq_list = list()
    seq_len_list = list()
    for i, (k, v) in enumerate(data_dict.items()):
        name_list.append(k)
        seq_list.append(v)
        seq_len_list.append(len(v))

    seq_max_len = max(seq_len_list)
    set_max_len = (seq_max_len // 80 + int(seq_max_len % 80 != 0)) * 80
    seq_encoding_list = list(map(lambda x: seq2encoding(x), seq_list))
    seq_encoding_pad_list = list(map(lambda x: padding(x,set_max_len), seq_encoding_list))
    data_fcn_2 = list(map(lambda x: get_data_fcn(x[0], x[1], set_max_len),
                          zip(seq_encoding_pad_list, seq_len_list)))

    seq_encoding_pad = torch.tensor(np.stack(seq_encoding_pad_list, axis=0)).float()
    data_fcn_2 = torch.tensor(np.stack(data_fcn_2, axis=0)).float()
    seq_length = torch.tensor(seq_len_list).long()
    tokens = generate_token_batch(alphabet, seq_list)
    return data_fcn_2, tokens, seq_encoding_pad, seq_length, name_list, set_max_len, seq_list, seq_len_list


def get_data_fcn(data_seq, data_length, set_length):
    perm = list(product(np.arange(4), np.arange(4)))
    data_fcn = np.zeros((16, set_length, set_length))
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_length, :data_length] = np.matmul(
            data_seq[:data_length, i].reshape(-1, 1),
            data_seq[:data_length, j].reshape(1, -1)
        )
    data_fcn_1 = np.zeros((1, set_length, set_length))
    data_fcn_1[0, :data_length, :data_length] = creatmat(data_seq[:data_length, :])
    data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)

    return data_fcn_2


def Gaussian(x):
    return math.exp(-0.5 * (x * x))


def paired(x, y):
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


# 产生RNA二级结构pair probability的算法
def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat


def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str) + int(alphabet.prepend_bos), ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens


def padding(data_array, maxlen):
    a, b = data_array.shape
    # np.pad(array, ((before_1,after_1),……,(before_n,after_n),module)
    return np.pad(data_array, ((0, maxlen - a), (0, 0)), 'constant')


def contact_map_masks(data_lens, matrix_rep):
    n_seq = len(data_lens)
    assert matrix_rep.shape[0] == n_seq
    for i in range(n_seq):
        l = int(data_lens[i].cpu().numpy())
        matrix_rep[i, :, :l, :l] = 1
    return matrix_rep


def get_model_prediction(args):
    model = DiffusionRNA2dPrediction(
        num_classes=args.num_classes,
        diffusion_dim=args.diffusion_dim,
        cond_dim=args.cond_dim,
        diffusion_steps=args.diffusion_steps,
        dp_rate=args.dp_rate,
        u_ckpt=args.u_conditioner_ckpt
    )
    alphabet = model.get_alphabet()
    return model, alphabet


def vote4struct(struc_list: List[np.ndarray]) -> np.ndarray:
    """
    Vote for the structure with the most votes.
    Args:
        struc_list: a list of predicted structures.

    Returns:
        The structure with the most votes.
    """
    id_struc_dict = dict()
    vote_dict = collections.defaultdict(int)

    for index, pred in enumerate(struc_list):
        id_loc = pred.argmax(axis=0)
        id_loc = list(id_loc)
        id_loc = ''.join(str(i) for i in id_loc)
        id_struc_dict[(index, id_loc)] = pred
        vote_dict[id_loc] += 1

    vote_id = max(vote_dict, key=vote_dict.get)

    for k, v in id_struc_dict.items():
        if k[1] == vote_id:
            return v


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    config = process_config(join(ROOT_PATH, 'config.json'))
    model, alphabet = get_model_prediction(config.model)
    data_fcn_2, tokens, seq_encoding_pad_list, seq_length, name_list, set_max_len = \
        get_data(join(ROOT_PATH + '/predict_data/' + config.predict_data), alphabet)


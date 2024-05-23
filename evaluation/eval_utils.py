# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
from os.path import join
from torch.utils.data import DataLoader
import json
import munch
from functools import partial
from datasets.data_generator import Dataset, diff_collate_fn
from models.model import DiffusionRNA2dPrediction
import collections
from typing import List
import copy
import pickle
from prettytable import PrettyTable

HOME_PATH = os.getcwd()

dataset_choices = ['RNAStrAlign', 'archiveII', 'bpRNA', 'bpRNAnew', 'pdbnew', 'TS_hard']
DATA_ROOT_PATH = '/home/students/zhyl_fyz/new_dataset/preprocess/'


def clean_dict(d, keys):
    d2 = copy.deepcopy(d)
    for key in keys:
        if key in d2:
            del d2[key]
    return d2


def log_eval_metrics(eval_dict):
    eval_metrics = {}
    if len(eval_metrics) == 0:
        for metric_name, metric_value in eval_dict.items():
            eval_metrics[metric_name] = [metric_value]
    else:
        for metric_name, metric_value in eval_dict.items():
            eval_metrics[metric_name].append(metric_value)
    return eval_metrics


def get_metric_table(metric_dict, count):
    table = PrettyTable()
    table.add_column('count', count)
    if len(metric_dict) > 0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values)
    return table


def save_metrics(test_metrics, save_path):
    # Save metrics
    with open(join(save_path, 'metrics_test.pickle'), 'wb') as f:
        pickle.dump(test_metrics, f)

    metric_table = get_metric_table(test_metrics, count=list(range(0, 1)))
    with open(join(save_path, 'metrics_test.txt'), 'w') as f:
        f.write(str(metric_table))


def parse_config(json_file):
    with open(join(HOME_PATH, json_file), 'r') as f:
        config_dict = json.load(f)

    config = munch.Munch(config_dict)
    config.model = munch.Munch(config.model)
    config.data = munch.Munch(config.data)
    return config


def get_data_test(args, alphabet):
    """
    Get the test data loader.
    Args:
        args: config.data
        alphabet: alphabet

    Returns:
        test_loader: test data loader
    """
    assert args.dataset in dataset_choices

    if args.dataset == 'RNAStrAlign':
        test = Dataset([join(DATA_ROOT_PATH, args.dataset, 'test')])

    elif args.dataset == 'archiveII':
        test = Dataset([join(DATA_ROOT_PATH, args.dataset)])

    elif args.dataset == 'bpRNA':
        test = Dataset([join(DATA_ROOT_PATH, args.dataset, 'TS0')])

    elif args.dataset == 'bpRNAnew':
        test = Dataset([join(DATA_ROOT_PATH, args.dataset, 'bpRNAnew')])

    # elif args.dataset == 'pdbnew':
    #     test = Dataset([join(DATA_ROOT_PATH, args.dataset, 'TS1'),
    #                       join(DATA_ROOT_PATH, args.dataset, 'TS2'),
    #                       join(DATA_ROOT_PATH, args.dataset, 'TS3')])
    elif args.dataset == 'pdbnew':
        test = Dataset([join(DATA_ROOT_PATH, args.dataset, 'TS3')])
    elif args.dataset == 'TS_hard':
        test = Dataset([join(DATA_ROOT_PATH, 'pdbnew', args.dataset)])

    else:
        raise NotImplementedError

    partial_collate_fn = partial(diff_collate_fn, alphabet=alphabet)

    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    return test_loader


def get_model_test(args):
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

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import pickle as cPickle
import collections
from os.path import join
import time

file_root_path = '~/new_dataset/'

save_root_path = file_root_path + 'binning/'

dataset_list = ['RNAStrAlign', 'bpRNA', 'bpRNAnew', 'pdbnew']

start_time = time.time()
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq seq_raw length name pairs')


def partition(array, low, high):
    pivot = array[high]

    i = low - 1
    for j in range(low, high):
        if array[j].length <= pivot.length:
            i += 1
            array[i], array[j] = array[j], array[i]

    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


def quick_sort_iterative(array):
    if len(array) <= 1:
        return array

    stack = []
    stack.append((0, len(array) - 1))

    while stack:
        low, high = stack.pop()

        pivot_index = partition(array, low, high)

        if low < pivot_index - 1:
            stack.append((low, pivot_index - 1))
        if pivot_index + 1 < high:
            stack.append((pivot_index + 1, high))

    return array


for dataset in dataset_list:
    print('#' * 10, f'Start binning {dataset}', '#' * 10)

    file_list = os.listdir(join(file_root_path, dataset))

    for file in file_list:
        name = 'bpRNA-pdb'
        print(f'start load {dataset} {name}')

        save_path = save_root_path + dataset + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file, 'rb') as f:
            LoadData = cPickle.load(f)

        sorted_data = quick_sort_iterative(LoadData)

        length_min = sorted_data[0].length
        length_max = sorted_data[-1].length
        sample_num = len(sorted_data)

        print(f'In {dataset} {name}, the min length：{length_min}')
        print(f'In {dataset} {name}, the max length：{length_max}')

        binned_data = list()
        set_len = 80
        set_interval = 80
        save_point = 1

        for i, data in enumerate(sorted_data):
            if set_len == 80:
                set_batch = 128
            elif set_len == 160:
                set_batch = 64
            elif set_len > 160 and set_len <= 320:
                set_batch = 16
            elif set_len > 320 and set_len <= 640:
                set_batch = 4
            elif set_len > 640 and set_len <= 1280:
                set_batch = 2
            else:
                set_batch = 1

            if data.length <= set_len:
                if len(binned_data) < set_batch:
                    binned_data.append(data)
                    if i == sample_num - 1:
                        with open(join(save_path, f'{name}_{set_len}_{save_point}.cPickle'), 'wb') as f:
                            cPickle.dump(binned_data, f)
                elif len(binned_data) == set_batch:
                    with open(join(save_path, f'{name}_{set_len}_{save_point}.cPickle'), 'wb') as f:
                        cPickle.dump(binned_data, f)
                    save_point += 1
                    binned_data = list()
                    binned_data.append(data)
                    if i == sample_num - 1:
                        with open(join(save_path, f'{name}_{set_len}_{save_point}.cPickle'), 'wb') as f:
                            cPickle.dump(binned_data, f)

            else:
                if len(binned_data):
                    with open(join(save_path, f'{name}_{set_len}_{save_point}.cPickle'), 'wb') as f:
                        cPickle.dump(binned_data, f)
                    save_point += 1

                while set_len < data.length:
                    set_len += set_interval
                binned_data = list()
                binned_data.append(data)
                if i == sample_num - 1:
                    with open(join(save_path, f'{name}_{set_len}_{save_point}.cPickle'), 'wb') as f:
                        cPickle.dump(binned_data, f)

        print(f'Time cost: {round(time.time() - start_time, 2)} seconds','\n',
              '-' * 10, f'Finish binning {dataset} {name}', '-' * 10, '\n')

print('#' * 10, 'Finish all binning', '#' * 10, '\n',
      f'Total time cost: {round(time.time() - start_time, 2)} seconds')

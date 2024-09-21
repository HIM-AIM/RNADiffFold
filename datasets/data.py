# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from .data_generator import Dataset, diff_collate_fn
from os.path import join
from functools import partial

dataset_choices = ['RNAStrAlign', 'archiveII', 'bpRNA', 'bpRNAnew', 'pdbnew', 'all']

ROOT_PATH = './data'


def add_data_args(parser):
    # Data params
    parser.add_argument('--dataset', type=str, default='bpRNA', choices=dataset_choices)
    parser.add_argument('--seq_len', type=str, default='160', choices={'160', '600', '640', 'all'})
    parser.add_argument('--upsampling', type=eval, default=False)

    # Train params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pin_memory', type=eval, default=False)


def get_data_id(args):
    return '{}_{}'.format(args.dataset, args.seq_len)


def get_data(args, alphabet):
    assert args.dataset in dataset_choices

    if args.dataset == 'RNAStrAlign':
        train = Dataset([join(ROOT_PATH, args.dataset, 'train')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'val')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'test')])

    elif args.dataset == 'bpRNA':
        train = Dataset([join(ROOT_PATH, args.dataset, 'TR0')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'VL0')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'TS0')])

    elif args.dataset == 'bpRNAnew':
        train = Dataset([join(ROOT_PATH, args.dataset, 'mutate')], upsampling=True)
        val = Dataset([join(ROOT_PATH, 'bpRNA', 'VL0')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'bpRNAnew')])
    elif args.dataset == 'pdbnew':
        train = Dataset([join(ROOT_PATH, args.dataset, 'TR1')], upsampling=True)
        val = Dataset([join(ROOT_PATH, args.dataset, 'VL1')])
        test = Dataset([join(ROOT_PATH, args.dataset, 'TS1'),
                        join(ROOT_PATH, args.dataset, 'TS2'),
                        join(ROOT_PATH, args.dataset, 'TS3')
                        ])

    elif args.dataset == 'all':
        train = Dataset([join(ROOT_PATH, 'RNAStrAlign/train'),
                         join(ROOT_PATH, 'bpRNA/TR0/'),
                         join(ROOT_PATH, 'bpRNAnew/mutate')], upsampling=True)
        val = Dataset([join(ROOT_PATH, 'bpRNA/VL0/'),
                       join(ROOT_PATH, 'RNAStrAlign/val')])
        test = Dataset([join(ROOT_PATH, 'bpRNA/TS0/'),
                        join(ROOT_PATH, 'RNAStrAlign/test'),
                        join(ROOT_PATH, 'bpRNAnew/bpRNAnew')])

    else:
        raise NotImplementedError

    partial_collate_fn = partial(diff_collate_fn, alphabet=alphabet)

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=partial_collate_fn,
                              pin_memory=args.pin_memory,
                              drop_last=True)

    val_loader = DataLoader(val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=partial_collate_fn,
                            pin_memory=args.pin_memory,
                            drop_last=False)

    test_loader = DataLoader(test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=partial_collate_fn,
                             pin_memory=args.pin_memory,
                             drop_last=False)

    return train_loader, val_loader, test_loader

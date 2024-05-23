# -*- coding: utf-8 -*-
import torch
import argparse
import collections

from common.utils import add_parent_path, set_seeds

add_parent_path(level=1)

from models.model import get_model, get_model_id, add_model_args
from optim.multistep import get_optim, get_optim_id, add_optim_args
from datasets.data import get_data_id, add_data_args, get_data
from experiment import Experiment, add_exp_args


# Setup
parser = argparse.ArgumentParser()

add_model_args(parser)
add_data_args(parser)
add_optim_args(parser)
add_exp_args(parser)

args = parser.parse_args()

set_seeds(args.seed)

# model
model_id = get_model_id(args)
model, alphabet = get_model(args)

# ckpt_path = '/home/students/zhyl_fyz/difffoldrna/ckpt/train.seed.2023.pt'
# checkpoint = torch.load(ckpt_path, map_location='cpu')
# model.load_state_dict(checkpoint['model'])
# print('load model from {}'.format(ckpt_path))

# data
data_id = get_data_id(args)
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
train_loader, val_loader, test_loader = get_data(args, alphabet)

# optimizer
optim_id = get_optim_id(args)
optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)

# training
exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()

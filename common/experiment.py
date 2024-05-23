# -*- coding: utf-8 -*-
import os
from os.path import join
import time
import pickle
import torch
from prettytable import PrettyTable
from common.utils import get_args_table, get_metric_table, clean_dict

from torch.utils.tensorboard import SummaryWriter
import wandb

import pathlib

HOME = str(pathlib.Path.home())


def add_exp_args(parser):
    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true', default=False)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_wandb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)


class EarlyStopping(object):
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0.0
        self.best_epoch = None
        self.counter = 0
        self.save_name = None
        self.early_stop = False
        self.ckpt_save = False

    def __call__(self, val_f1_score, current_epoch):
        if val_f1_score > self.best_score:
            self.best_score = val_f1_score
            self.best_epoch = current_epoch
            self.ckpt_save = True
            self.save_name = f'best_checkpoint_{self.best_epoch + 1}.pt'
            self.counter = 0

        elif val_f1_score - self.best_score < self.min_delta:
            self.ckpt_save = False
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


class BaseExperiment(object):

    def __init__(self,
                 model,
                 optimizer,
                 scheduler_iter,
                 scheduler_epoch,
                 log_path,
                 eval_every,
                 check_every,
                 save_name=None):

        # Objects
        self.model = model
        self.optimizer = optimizer
        self.scheduler_iter = scheduler_iter
        self.scheduler_epoch = scheduler_epoch

        # Paths
        self.log_path = log_path
        self.check_path = os.path.join(log_path, 'check')

        # Intervals
        self.eval_every = eval_every
        self.check_every = check_every
        self.save_name = save_name

        # Initialize
        self.current_epoch = 0
        self.train_metrics = {}
        self.eval_metrics = {}
        self.eval_epochs = []
        self.test_metrics = {}

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0)

    def train_fn(self, epoch):
        raise NotImplementedError()

    def val_fn(self, epoch):
        raise NotImplementedError()

    def test_fn(self, epoch):
        raise NotImplementedError()

    def log_fn(self, epoch, train_dict, val_dict, test_dict):
        raise NotImplementedError()

    def log_train_metrics(self, train_dict):
        if len(self.train_metrics) == 0:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in train_dict.items():
                self.train_metrics[metric_name].append(metric_value)

    def log_eval_metrics(self, eval_dict):
        if len(self.eval_metrics) == 0:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in eval_dict.items():
                self.eval_metrics[metric_name].append(metric_value)

    def log_test_metrics(self, test_dict):
        if test_dict is not None:
            if len(self.test_metrics) == 0:
                for metric_name, metric_value in test_dict.items():
                    self.test_metrics[metric_name] = [metric_value]
            else:
                for metric_name, metric_value in test_dict.items():
                    self.test_metrics[metric_name].append(metric_value)
        else:
            print('test_dict is empty!')

    def create_folders(self):
        # Create log folder
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        print("Storing logs in:", self.log_path)

        # Create check folder
        if self.check_every is not None and not os.path.exists(self.check_path):
            os.makedirs(self.check_path)
            print("Storing checkpoints in:", self.check_path)

    def save_args(self, args):

        # Save args
        with open(join(self.log_path, 'args.pickle'), "wb") as f:
            pickle.dump(args, f)

        # Save args table
        args_table = get_args_table(vars(args))
        with open(join(self.log_path, 'args_table.txt'), "w") as f:
            f.write(str(args_table))

    def save_metrics(self):

        # Save metrics
        with open(join(self.log_path, 'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(join(self.log_path, 'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)
        with open(join(self.log_path, 'metrics_test.pickle'), 'wb') as f:
            pickle.dump(self.test_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(self.train_metrics, epochs=list(range(1, self.current_epoch + 2)))
        with open(join(self.log_path, 'metrics_train.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.eval_metrics, epochs=[e + 1 for e in self.eval_epochs])
        with open(join(self.log_path, 'metrics_eval.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.test_metrics, epochs=[self.current_epoch + 1])
        with open(join(self.log_path, 'metrics_test.txt'), "w") as f:
            f.write(str(metric_table))

    def checkpoint_save(self, name='checkpoint.pt'):
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'model': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None}
        torch.save(checkpoint, os.path.join(self.check_path, name))

    def checkpoint_load(self, check_path, name='checkpoint.pt'):
        checkpoint = torch.load(os.path.join(check_path, name))
        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

    def run(self, epochs):
        for epoch in range(self.current_epoch, epochs):

            # Train
            train_dict = self.train_fn(epoch)
            self.log_train_metrics(train_dict)

            # val
            if (epoch + 1) % self.eval_every == 0:
                val_dict = self.val_fn(epoch)
                self.early_stopping(val_dict['f1'], epoch)
                if self.early_stopping.ckpt_save:
                    self.checkpoint_save(name=self.early_stopping.save_name)
                self.log_eval_metrics(val_dict)
                self.eval_epochs.append(epoch)
            else:
                val_dict = None

            # test
            if (epoch + 1) == epochs:
                if self.early_stopping.save_name is not None:
                    self.checkpoint_load(self.check_path, name=self.early_stopping.save_name)
                    print(f'load best checkpoint:{self.early_stopping.save_name}')
                else:
                    print('load last checkpoint')
                test_dict, f1_pre_rec_df = self.test_fn(epoch)
                f1_pre_rec_df.to_csv(join(self.log_path, f'{self.save_name}.csv'), index=False, header=False)
                self.log_test_metrics(test_dict)
            elif self.early_stopping.early_stop:
                print('Early stopping')
                if self.early_stopping.save_name is not None:
                    self.checkpoint_load(self.check_path, name=self.early_stopping.save_name)
                    print(f'load best checkpoint:{self.early_stopping.save_name}')
                else:
                    print('load last checkpoint')
                test_dict, f1_pre_rec_df = self.test_fn(epoch)
                f1_pre_rec_df.to_csv(join(self.log_path, f'{self.save_name}.csv'), index=False, header=True)
                self.log_test_metrics(test_dict)
                # Log
                self.save_metrics()
                self.log_fn(epoch, train_dict, val_dict, test_dict)
                break
            else:
                test_dict = None

            # Log
            self.save_metrics()
            self.log_fn(epoch, train_dict, val_dict, test_dict)

            # Checkpoint
            self.current_epoch += 1
            if (epoch + 1) % self.check_every == 0:
                self.checkpoint_save()


class DiffusionExperiment(BaseExperiment):
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel',
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, val_loader, test_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):

        if args.log_home is None:
            self.log_base = join(HOME, 'logs', 'rnadifffold')
        else:
            self.log_base = join(args.log_home, 'logs', 'rnadifffold')

        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.project is None:
            args.project = 'rnadifffold'

        save_name = f'{args.name}.{args.dataset}.seed_{args.seed}.{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        model.to(args.device)

        super(DiffusionExperiment, self).__init__(model=model,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=join(self.log_base, f'{data_id}_{model_id}_{optim_id}',
                                                                args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every,
                                                  save_name=save_name)

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # dry run
        self.dry_run = args.dry_run

        # Init logging
        if not self.dry_run:
            args_dict = clean_dict(vars(args), keys=self.no_log_keys)
            if args.log_tb:
                self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
                self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
            if args.log_wandb:
                wandb.init(config=args_dict, project=args.project, id=args.name, dir=self.log_path)

    def log_fn(self, epoch, train_dict, val_dict, test_dict):

        if not self.dry_run:
            # Tensorboard
            if self.args.log_tb:
                for metric_name, metric_value in train_dict.items():
                    self.writer.add_scalar('train/{}'.format(metric_name), metric_value, global_step=epoch+1)

                if val_dict:
                    for metric_name, metric_value in val_dict.items():
                        self.writer.add_scalar('val/{}'.format(metric_name), metric_value, global_step=epoch+1)

                if test_dict:
                    for metric_name, metric_value in test_dict.items():
                        self.writer.add_scalar('test/{}'.format(metric_name), metric_value, global_step=epoch+1)


            # Weights & Biases
            if self.args.log_wandb:
                for metric_name, metric_value in train_dict.items():
                    wandb.log({'train/{}'.format(metric_name): metric_value}, step=epoch + 1)
                if val_dict:
                    for metric_name, metric_value in val_dict.items():
                        wandb.log({'val/{}'.format(metric_name): metric_value}, step=epoch + 1)
                if test_dict:
                    metric_name_list = []
                    metric_value_list = []
                    for metric_name, metric_value in test_dict.items():
                        metric_name_list.append(metric_name)
                        metric_value_list.append(metric_value)
                        table = wandb.Table(columns=metric_name_list, data=[metric_value_list])
                        wandb.log({'test': table})
        else:
            pass

    def resume(self):
        resume_path = os.path.join(self.log_base, f'{self.data_id}_{self.model_id}_{self.optim_id}', self.args.resume,
                                   'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]

            if epoch in self.eval_epochs:
                val_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    val_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else:
                val_dict = None

            if (epoch + 1) == self.args.epochs:
                test_dict = {}
                for metric_name, metric_values in self.test_metrics.items():
                    test_dict[metric_name] = metric_values[epoch]
            else:
                test_dict = None

            self.log_fn(epoch, train_dict=train_dict, val_dict=val_dict, test_dict=test_dict)

    def run(self):
        if self.args.resume:
            self.resume()
        super(DiffusionExperiment, self).run(epochs=self.args.epochs)


class DataParallelDistribution(torch.nn.DataParallel):
    """
    A DataParallel wrapper for Distribution.
    To be used instead of nn.DataParallel for Distribution objects.
    """

    def log_prob(self, *args, **kwargs):
        return self.forward(*args, mode='log_prob', **kwargs)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)

    def sample_with_log_prob(self, *args, **kwargs):
        return self.module.sample_with_log_prob(*args, **kwargs)

# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

from common.utils import add_parent_path
from common.experiment import add_exp_args as add_exp_args_parent
from common.experiment import DiffusionExperiment
from common.data_utils import contact_map_masks
from common.loss_utils import bce_loss, evaluate_f1_precision_recall
from common.loss_utils import calculate_auc, calculate_mattews_correlation_coefficient,rna_evaluation
add_parent_path(level=2)


def add_exp_args(parser):
    add_exp_args_parent(parser)


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        device = self.args.device
        for _, (contact, data_fcn_2, data_seq_raw, data_length, _, set_max_len, data_seq_encoding) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            contact = contact.to(device)
            data_fcn_2 = data_fcn_2.to(device)
            matrix_rep = torch.zeros_like(contact)
            data_length = data_length.to(device)
            data_seq_raw = data_seq_raw.to(device)
            data_seq_encoding = data_seq_encoding.to(device)
            contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

            loss = self.model(contact, data_fcn_2, data_seq_raw, contact_masks, set_max_len, data_seq_encoding)
            loss.backward()

            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(contact)
            loss_count += len(contact)
            print('Training. Epoch: {}/{}, Bits/dim: {:.5f}'.
                  format(epoch + 1, self.args.epochs, loss_sum / loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum / loss_count}

    def val_fn(self, epoch):
        self.model.eval()

        device = self.args.device
        with torch.no_grad():
            loss_count = 0
            val_loss_sum = 0.0
            auc_score = 0.0
            auc_count = 0
            val_no_train = list()
            mcc_no_train = list()

            for _, (contact, data_fcn_2, data_seq_raw, data_length, _, set_max_len, data_seq_encoding) in enumerate(self.val_loader):
                data_fcn_2 = data_fcn_2.to(device)
                matrix_rep = torch.zeros_like(contact)
                data_length = data_length.to(device)
                data_seq_raw = data_seq_raw.to(device)
                data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                batch_size = contact.shape[0]
                pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)

                pred_x0 = pred_x0.cpu().float()
                val_loss_sum += bce_loss(pred_x0.float(), contact.float()).cpu().item()
                loss_count += len(contact)
                auc_score += calculate_auc(contact.float(), pred_x0)
                auc_count += 1
                val_no_train_tmp = list(map(lambda i: evaluate_f1_precision_recall(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                val_no_train += val_no_train_tmp

                mcc_no_train_tmp = list(map(lambda i: calculate_mattews_correlation_coefficient(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                mcc_no_train += mcc_no_train_tmp

            val_precision, val_recall, val_f1 = zip(*val_no_train)

            val_precision = np.average(np.nan_to_num(np.array(val_precision)))
            val_recall = np.average(np.nan_to_num(np.array(val_recall)))
            val_f1 = np.average(np.nan_to_num(np.array(val_f1)))

            mcc_final = np.average(np.nan_to_num(np.array(mcc_no_train)))

            print('#' * 80)
            print('Average val F1 score: ', round(val_f1, 3))
            print('Average val precision: ', round(val_precision, 3))
            print('Average val recall: ', round(val_recall, 3))
            print('#' * 80)
            print('Average val MCC', round(mcc_final, 3))
            print('#' * 80)
            print('')
        return {'f1': val_f1, 'precision': val_precision, 'recall': val_recall,
                'auc_score': auc_score / auc_count, 'mcc': mcc_final, 'bce_loss': val_loss_sum / loss_count}

    def test_fn(self, epoch):
        self.model.eval()
        device = self.args.device
        with torch.no_grad():
            test_no_train = list()
            total_name_list = list()
            total_length_list = list()

            for _, (contact, data_fcn_2, data_seq_raw, data_length, data_name, set_max_len, data_seq_encoding) in enumerate(
                    self.test_loader):
                total_name_list += [item for item in data_name]
                total_length_list += [item.item() for item in data_length]

                data_fcn_2 = data_fcn_2.to(device)
                matrix_rep = torch.zeros_like(contact)
                data_length = data_length.to(device)
                data_seq_raw = data_seq_raw.to(device)
                data_seq_encoding = data_seq_encoding.to(device)
                contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

                # calculate contact loss
                batch_size = contact.shape[0]
                pred_x0, _ = self.model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)

                pred_x0 = pred_x0.cpu().float()

                test_no_train_tmp = list(map(lambda i: rna_evaluation(
                    pred_x0[i].squeeze(), contact.float()[i].squeeze()), range(pred_x0.shape[0])))
                test_no_train += test_no_train_tmp

            accuracy, prec, recall, sens, spec, F1, MCC = zip(*test_no_train)

            f1_pre_rec_df = pd.DataFrame({'name': total_name_list,
                                          'length': total_length_list,
                                          'accuracy': list(np.array(accuracy)),
                                          'precision': list(np.array(prec)),
                                          'recall': list(np.array(recall)),
                                          'sensitivity': list(np.array(sens)),
                                          'specificity': list(np.array(spec)),
                                          'f1': list(np.array(F1)),
                                          'mcc': list(np.array(MCC))})

            accuracy = np.average(np.nan_to_num(np.array(accuracy)))
            precision = np.average(np.nan_to_num(np.array(prec)))
            recall = np.average(np.nan_to_num(np.array(recall)))
            sensitivity = np.average(np.nan_to_num(np.array(sens)))
            specificity = np.average(np.nan_to_num(np.array(spec)))
            F1 = np.average(np.nan_to_num(np.array(F1)))
            MCC = np.average(np.nan_to_num(np.array(MCC)))

            print('#' * 40)
            print('Average testing accuracy: ', round(accuracy, 3))
            print('Average testing F1 score: ', round(F1, 3))
            print('Average testing precision: ', round(precision, 3))
            print('Average testing recall: ', round(recall, 3))
            print('Average testing sensitivity: ', round(sensitivity, 3))
            print('Average testing specificity: ', round(specificity, 3))
            print('#' * 40)
            print('Average testing MCC', round(MCC, 3))
            print('#' * 40)
            print('')
        return {'f1': F1, 'precision': precision, 'recall': recall,
                'sensitivity': sensitivity, 'specificity': specificity, 'accuracy': accuracy, 'mcc': MCC}, f1_pre_rec_df

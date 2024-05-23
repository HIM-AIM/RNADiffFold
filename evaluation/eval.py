# -*- coding: utf-8 -*-

import time
import os
from os.path import join
import torch
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from eval_utils import parse_config, get_data_test, get_model_test, vote4struct, clean_dict, log_eval_metrics, \
    save_metrics
from common.data_utils import contact_map_masks
from common.loss_utils import rna_evaluation
import collections


def evaluation(args, eval_model, dataloader):
    eval_model.eval()
    device = args.device

    with torch.no_grad():
        test_no_train = list()
        total_name_list = list()
        total_length_list = list()
        for _, (contact, data_fcn_2, data_seq_raw, data_length, data_name, set_max_len, data_seq_encoding) in enumerate(dataloader):
            total_name_list += [item for item in data_name]
            total_length_list += [item.item() for item in data_length]

            data_fcn_2 = data_fcn_2.to(device)
            matrix_rep = torch.zeros_like(contact)
            data_length = data_length.to(device)
            data_seq_raw = data_seq_raw.to(device)
            data_seq_encoding = data_seq_encoding.to(device)
            contact_masks = contact_map_masks(data_length, matrix_rep).to(device)
            batch_size = contact.shape[0]

            # for multi conformations sampling
            pred_x0_copy_dict = dict()
            best_pred_x0_i_list = list()
            candidate_seeds = np.arange(0, 2023)
            select_seeds = np.random.choice(candidate_seeds, args.num_samples).tolist()
            for seed_ind in select_seeds:
                torch.manual_seed(seed_ind)

                pred_x0, _ = eval_model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)
                pred_x0_copy_dict[seed_ind] = pred_x0

            for i in tqdm(range(pred_x0.shape[0]), desc=f'vote for the most common structure', total=pred_x0.shape[0]):
                pred_x0_i_list = [pred_x0_copy_dict[num_copy][i].squeeze().cpu().numpy() for num_copy in select_seeds]
                best_pred_x0_i = torch.Tensor(vote4struct(pred_x0_i_list))
                best_pred_x0_i_list.append(best_pred_x0_i)

            pred_x0 = torch.stack(best_pred_x0_i_list, dim=0)
            pred_x0 = pred_x0.cpu().float().unsqueeze(1)
            # pred_x0, _ = eval_model.sample(batch_size, data_fcn_2, data_seq_raw, set_max_len, contact_masks, data_seq_encoding)
            # pred_x0 = pred_x0.cpu().float()

            # test_loss_sum += bce_loss(pred_x0, contact.float()).cpu().item()
            # loss_count += len(contact)
            # auc_score += calculate_auc(contact.float(), pred_x0)
            # auc_count += 1

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


if __name__ == "__main__":
    start = time.time()
    # config
    config = parse_config('config.json')
    torch.manual_seed(config.seed)
    print('#'*10, f'Start evaluate {config.data.dataset}', '#'*10)
    save_root_path = config.save_root_path
    name = f'{config.project_name}.round_{config.round}.dataset_{config.data.dataset}.num_sample_{config.num_samples}'
    save_path = join(config.save_root_path, 'results', f'dataset_{config.data.dataset}', f'round_{config.round}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model, alphabet = get_model_test(config.model)
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'contact data_fcn_2 seq_raw length name')
    test_loader = get_data_test(config.data, alphabet)

    # model load checkpoint
    print(f"Load model checkpoint from: {config.model_ckpt_path}")
    checkpoint = torch.load(config.model_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(config.device)

    # config_dict = clean_dict(vars(config.toDict()), keys=no_log_keys)

    if not config.dry_run:
        wandb.init(project=config.project_name, name=name, config=config.toDict(), dir=save_path)

    test_dict, result_df = evaluation(config, model, test_loader)

    result_df.to_csv(join(save_path, f'{name}.csv'), index=False, header=True)
    test_metrics = log_eval_metrics(test_dict)
    if not config.dry_run:
        save_metrics(test_metrics, save_path)
        metric_name_list = []
        metric_value_list = []
        for metric_name, metric_value in test_dict.items():
            metric_name_list.append(metric_name)
            metric_value_list.append(metric_value)
            table = wandb.Table(columns=metric_name_list, data=[metric_value_list])
            wandb.log({'test': table})

    stop_time = time.time()
    print(f'Finished in {(stop_time - start) / 60:.2f} minutes')
    print(f'Finish time: {time.asctime()}')

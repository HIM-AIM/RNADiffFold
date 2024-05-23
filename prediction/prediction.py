# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from prediction_utils import *
from tqdm import tqdm


def prediction(config, model, data_fcn_2, tokens, seq_encoding_pad, seq_length, set_max_len):
    device = config.device
    model.to(device)
    model.eval()
    with torch.no_grad():
        data_fcn_2 = data_fcn_2.to(device)
        tokens = tokens.to(device)
        seq_encoding_pad = seq_encoding_pad.to(device)
        seq_length = seq_length.to(device)
        batch_size = data_fcn_2.shape[0]
        matrix_rep = torch.zeros((batch_size, set_max_len, set_max_len)).unsqueeze(1)
        contact_masks = contact_map_masks(seq_length, matrix_rep).to(device)

        # for multi conformations sampling
        pred_x0_copy_dict = dict()
        best_pred_x0_i_list = list()
        candidate_seeds = np.arange(1970, 2023)
        select_seeds = np.random.choice(candidate_seeds, config.num_samples).tolist()
        for seed_ind in select_seeds:
            torch.manual_seed(seed_ind)

            pred_x0, _ = model.sample(batch_size, data_fcn_2, tokens,
                                      set_max_len, contact_masks, seq_encoding_pad)
            pred_x0_copy_dict[seed_ind] = pred_x0

        for i in tqdm(range(pred_x0.shape[0]), desc=f'vote for the most common structure', total=pred_x0.shape[0]):
            pred_x0_i_list = [pred_x0_copy_dict[num_copy][i].squeeze().cpu().numpy() for num_copy in select_seeds]
            best_pred_x0_i = torch.Tensor(vote4struct(pred_x0_i_list))
            best_pred_x0_i_list.append(best_pred_x0_i)

    return best_pred_x0_i_list


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    save_path = join(ROOT_PATH, 'predict_results/visualization')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = process_config(join(ROOT_PATH, 'config.json'))

    set_seed(config.seed)

    model, alphabet = get_model_prediction(config.model)

    # model load checkpoint
    print(f"Load model checkpoint from: {config.model_ckpt_path}")
    checkpoint = torch.load(config.model_ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # file_list = os.listdir(join(ROOT_PATH, 'predict_data/multi_conformation'))
    # for file in file_list:
    data_fcn_2, tokens, seq_encoding_pad, seq_length, name_list, set_max_len, seq_list, seq_len_list = \
        get_data(join(ROOT_PATH, f'predict_data/{config.predict_data}'), alphabet)

    predict_results_list = prediction(config, model, data_fcn_2, tokens, seq_encoding_pad, seq_length, set_max_len)
    pred_results_numpy_list = [pred.cpu().numpy() for pred in predict_results_list]

    for i, name in enumerate(name_list):
        df = contact2ct(pred_results_numpy_list[i], seq_list[i], seq_len_list[i])

        df.to_csv(join(save_path, f'{name}.ct'), sep='\t', index=False, header=False)
        print(f"Predicted contact map for {name} saved to {save_path}")



# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from models.diffusion_multinomial import MultinomialDiffusion
from models.layers import SegmentationUnet2DCondition
from models.condition.u_conditioner import Unet_conditioner
from models.condition.fm_conditioner.pretrained import load_model_and_alphabet_local
import lightning.pytorch as pl

CH_FOLD = 1
cond_ckpt_path = '../ckpt/cond_ckpt'


def add_model_args(parser):
    # Model params
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--diffusion_dim', type=int, default=8)
    parser.add_argument('--cond_dim', type=int, default=1)
    parser.add_argument('--dp_rate', type=float, default=0.)
    parser.add_argument('--u_conditioner_ckpt', type=str, default='ufold_train_alldata.pt')


def get_model_id(args):
    return 'multinomial_diffusion'


class DiffusionRNA2dPrediction(nn.Module):
    def __init__(self,
                 num_classes,
                 diffusion_dim,
                 cond_dim,
                 diffusion_steps,
                 dp_rate,
                 u_ckpt
                 ):
        super(DiffusionRNA2dPrediction, self).__init__()

        self.num_classes = num_classes
        self.diffusion_dim = diffusion_dim
        self.cond_dim = cond_dim
        self.diffusion_steps = diffusion_steps
        self.dp_rate = dp_rate
        self.u_ckpt = u_ckpt

        # condition
        self.fm_conditioner, self.alphabet = load_model_and_alphabet_local(
            join(cond_ckpt_path, 'RNA-FM_pretrained.pth'))
        self.u_conditioner = None
        self.load_u_conditioner()

        self.denoise_layer = SegmentationUnet2DCondition(
            num_classes=self.num_classes,
            dim=self.diffusion_dim,
            cond_dim=self.cond_dim,
            num_steps=self.diffusion_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=self.dp_rate
        )

        self.diffusion = MultinomialDiffusion(
            self.num_classes,
            self.diffusion_steps,
            self.denoise_layer
        )

    def load_u_conditioner(self):
        self.u_conditioner = Unet_conditioner(img_ch=17, output_ch=1)
        self.u_conditioner.load_state_dict(torch.load(join(cond_ckpt_path, self.u_ckpt), map_location='cpu'))
        condition_out = nn.Conv2d(int(32 * CH_FOLD), self.cond_dim, kernel_size=1, stride=1, padding=0)
        self.u_conditioner.Conv_1x1 = condition_out
        self.u_conditioner.requires_grad_(True)

    def get_alphabet(self):
        return self.alphabet

    # @torch.no_grad()
    def get_fm_embedding(self, data_seq_raw, set_max_len):
        self.fm_conditioner.eval()

        device = data_seq_raw.device

        fm_condition = dict()

        with torch.no_grad():
            backbone_result = self.fm_conditioner(data_seq_raw, need_head_weights=False, repr_layers=[12],
                                                  return_contacts=True)
            fm_embedding = backbone_result['representations'][12]
            fm_embedding = fm_embedding[:, 1:-1, :]

            fm_attention_map = backbone_result['attentions']
            b, l, n, l1, l2 = fm_attention_map.shape
            fm_attention_map = fm_attention_map.reshape(b, l*n, l1, l2)[:, :, 1:-1, 1:-1]

            padding_value = 0
            padding_size = (0, set_max_len - fm_attention_map.shape[-2], 0, set_max_len - fm_attention_map.shape[-1])
            fm_embedding_pad = torch.zeros(fm_embedding.shape[0], set_max_len - fm_embedding.shape[1],
                                           fm_embedding.shape[2]).to(device)
            fm_embedding = torch.cat([fm_embedding, fm_embedding_pad], dim=1)

            fm_attention_map = F.pad(fm_attention_map, padding_size, 'constant', value=padding_value)

            fm_condition['fm_embedding'] = fm_embedding
            fm_condition['fm_attention_map'] = fm_attention_map

        return fm_condition

    def get_ufold_condition(self, data_fcn_2):

        u_condition = self.u_conditioner(data_fcn_2)

        return u_condition

    def forward(self,
                x_0,
                data_fcn_2,
                data_seq_raw,
                contact_masks,
                set_max_len,
                data_seq_encoding
                ):

        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        loss = self.diffusion(x_0, fm_condition, u_condition, contact_masks, data_seq_encoding)

        loglik_bpd = -loss.sum()/(math.log(2) * x_0.shape.numel())
        return loglik_bpd

    @torch.no_grad()
    def sample(self,
               num_samples,
               data_fcn_2,
               data_seq_raw,
               set_max_len,
               contact_masks,
               seq_encoding
               ):
        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        pred_x_0, model_prob = self.diffusion.sample(
            num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding
        )

        return pred_x_0, model_prob

    @torch.no_grad()
    def sample_chain(self,
                     num_samples,
                     data_fcn_2,
                     data_seq_raw,
                     set_max_len,
                     contact_masks,
                     seq_encoding
                     ):
        fm_condition = self.get_fm_embedding(data_seq_raw, set_max_len)

        u_condition = self.get_ufold_condition(data_fcn_2)

        pred_x_0_chain, model_prob_chain, pred_x_0, model_prob = self.diffusion.sample_chain(
            num_samples, fm_condition, u_condition, contact_masks, set_max_len, seq_encoding
        )
        return pred_x_0_chain, model_prob_chain, pred_x_0, model_prob


def get_model(args):
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

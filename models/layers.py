# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from inspect import isfunction


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample_new(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample_SP_conv(nn.Module):
    """
    https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    named SP-conv in the paper, but basically a pixel unshuffle
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.downsample = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1=2, s2=2),
            nn.Conv2d(dim * 4, dim_out, 1)
        )

    def forward(self, x):
        return self.downsample(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000, flip_sin_to_cos=False):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim: int, scale: float = 1.0, flip_sin_to_cos=False):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(self.half_dim) * scale)
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        if self.flip_sin_to_cos:
            fouriered = torch.cat([fouriered[:, self.half_dim:], fouriered[:, :self.half_dim]], dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        # fouriered_dim=dim+1
        return fouriered


class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups=8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            *,
            time_emb_dim=None,
            cond_dim=None,
            groups=8,
    ):
        super().__init__()

        self.time_mlp = None
        if exists(time_emb_dim):
            self.time_mlp = nn.Sequential(
                Mish(),
                nn.Linear(time_emb_dim, dim_out)
            )

        self.cond_mlp = None
        if exists(cond_dim):
            self.cond_mlp = nn.Sequential(
                Mish(),
                nn.Linear(cond_dim, dim_out)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')

        h += time_emb

        if exists(self.cond_mlp) and exists(cond):
            cond = rearrange(cond, 'b c h w -> b h w c')
            cond = self.cond_mlp(cond)
            cond = rearrange(cond, 'b h w c -> b c h w')
            h += cond

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SegmentationUnet2DCondition(nn.Module):

    def __init__(
            self,
            num_classes,
            dim,
            cond_dim,
            num_steps,
            dim_mults=(1, 2, 4, 8),
            dropout=0.,
            learned_time_emb=True,
            cat_cond=True,
            scale_skip_connection=False
    ):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.embedding = nn.Embedding(num_classes, dim)

        self.dim = dim
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.cat_cond = cat_cond
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        self.dropout = nn.Dropout(p=dropout)

        if learned_time_emb:
            self.time_pos_emb = LearnedSinusoidalPosEmb(dim, scale=1.0, flip_sin_to_cos=False)
        else:
            self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps, rescale_steps=4000, flip_sin_to_cos=False)

        # condition
        self.to_time_cond = nn.Sequential(
            self.time_pos_emb,
            nn.Linear(self.dim + 1 if learned_time_emb else self.dim, 4 * self.dim),
            Mish(),
            nn.Linear(4 * self.dim, self.dim)
        )

        if self.cat_cond:
            self.to_cond = nn.Sequential(
                nn.Linear(2 * self.dim + self.cond_dim, 4 * self.dim),
                Mish(),
                nn.Linear(4 * self.dim, self.dim)
            )
        else:
            self.to_cond = nn.Sequential(
                nn.Linear(self.cond_dim, 4 * self.cond_dim),
                Mish(),
                nn.Linear(4 * self.cond_dim, self.cond_dim)
            )

        self.fm_cond_1 = nn.Sequential(
            nn.Linear(640, 64),
            Mish(),
            nn.Linear(64, 8)
        )

        self.fm_cond_2 = nn.Sequential(
            nn.Linear(240, 64),
            Mish(),
            nn.Linear(64, 8)
        )

        self.fm_cond = nn.Sequential(
            nn.Linear(16, 48),
            Mish(),
            nn.Linear(48, 8)
        )

        self.x_mlp = nn.Sequential(
            nn.Linear(48, 64),
            Mish(),
            nn.Linear(64, self.dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample_SP_conv(dim_out) if not is_last else nn.Identity(),
                Downsample_SP_conv(self.cond_dim) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_blocks1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_out, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8),
                Residual(Rezero(LinearAttention(dim_out))),
                Upsample_new(dim_out) if not is_last else nn.Identity(),
                ResnetBlock(dim_out, dim_in, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8)
            ]))

        out_dim = num_classes
        self.res_conv = ResnetBlock(dim, dim, time_emb_dim=dim, cond_dim=self.cond_dim, groups=8)
        self.out_conv = nn.Conv2d(dim, out_dim, 1)

    def forward(
            self,
            time,
            x,
            fm_condition,
            u_condition,
            seq_encoding
    ):
        x_shape = x.shape[1:]
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.size()

        x = self.embedding(x)
        assert x.shape == (B, C, H, W, self.dim)
        x = x.permute(0, 1, 4, 2, 3)
        assert x.shape == (B, C, self.dim, H, W)

        x = x.reshape(B, C * self.dim, H, W)

        cond = None

        fm_embedding = self.fm_cond_1(fm_condition['fm_embedding']).permute(0, 2, 1)
        fm_attention_map = self.fm_cond_2(fm_condition['fm_attention_map'].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cond_L = fm_embedding.size(-1)

        fm_out_cat = torch.cat([fm_embedding.unsqueeze(-1).repeat(1, 1, 1, cond_L),
                                fm_embedding.unsqueeze(-2).repeat(1, 1, cond_L, 1)], dim=1)
        seq_encoding = seq_encoding.permute(0, 2, 1)
        seq_out_cat = torch.cat([seq_encoding.unsqueeze(-1).repeat(1, 1, 1, cond_L),
                                 seq_encoding.unsqueeze(-2).repeat(1, 1, cond_L, 1)], dim=1)

        x = self.x_mlp(torch.cat([x,
                                  fm_out_cat,
                                  fm_attention_map,
                                  seq_out_cat,
                                  u_condition], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        t = self.to_time_cond(time)

        hiddens = []
        conds = []

        for ind, (resnet1, resnet2, attn, downsample, cond_downsample) in enumerate(self.downs):
            x = resnet1(x, t, cond)
            x = self.dropout(x)
            x = resnet2(x, t, cond)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x)
            if not self.cat_cond:
                if ind != len(self.downs) - 1:
                    conds.append(cond)
                cond = cond_downsample(cond)

        x = self.mid_blocks1(x, t, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cond)

        for ind, (resnet1, attn, upsample, resnet2, resnet3) in enumerate(self.ups):
            x = torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)
            x = resnet1(x, t, cond)
            x = attn(x)
            x = upsample(x)
            if not self.cat_cond:
                cond = conds.pop() if ind != len(self.ups) - 1 else cond
            x = resnet2(x, t, cond)
            x = resnet3(x, t, cond)

        # convert xt to onehot
        x = self.res_conv(x, t, cond)
        final = self.out_conv(x).view(B, self.num_classes, *x_shape)
        # make output matrix symmetric
        return torch.transpose(final, -1, -2) * final


from thop import profile

if __name__ == '__main__':
    x = torch.randint(0, 2, [2, 1, 160, 160])
    t_emb = torch.randint(0, 1000, [2])
    u_cond = torch.randn([2, 8, 160, 160])
    fm_cond = torch.randn([2, 160, 640])
    print(len(fm_cond))
    model = SegmentationUnet2DCondition(2, 8, 8, 1000, learned_time_emb=True, cat_cond=True)
    # out = model(x, t_emb, cond)
    # print(out)
    flops, params = profile(model, inputs=(t_emb, x, fm_cond, u_cond))
    print(flops, params)

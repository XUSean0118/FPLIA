import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import models
from utils import make_coord, make_coord_cell

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        self.in_dim = in_dim
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == "linear":
        return LinearAttention(in_channels, heads=1, dim_head=in_channels)
    else:
        raise NotImplementedError


class Encoder(nn.Module):
    def __init__(self, in_channels, ch, z_channels, double_z=False, ch_mult=(1,2,4,8), num_res_blocks=2, 
                 dropout=0.0, resamp_with_conv=True, attn_type="vanilla", ckpt_path=None, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif 'model' in ckpt:
            sd = torch.load(path, map_location="cpu")["model"]['sd']
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    

class Decoder(nn.Module):
    def __init__(self, ch, z_channels, ch_mult=(1,2,4,8), num_res_blocks=2,  
                 dropout=0.0, resamp_with_conv=True, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            self.up.insert(0, up) # prepend to get consistent order

        self.out_dim = block_out

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)

        return h

class LIA(nn.Module):
    def __init__(
        self,
        imnet_spec,
        pb_spec,
        fid_dim=None,
        prec_dim=None,
        fid=True,
        perc=False,
        perc_fid=False,
        fid_perc=False,
        blend=False,
        base_dim=256,
        head=8,
    ):
        super().__init__()
        self.dim = base_dim
        self.head = head
        self.fid = fid
        self.perc = perc
        self.perc_fid = perc_fid
        self.fid_perc = fid_perc
        self.blend = blend
        self.temperature = 1

        self.pb_encoder = models.make(pb_spec, args={'head': self.head})
        self.imnet_de = models.make(imnet_spec, args={'in_dim': self.dim})

        self.r = 3
        self.r_area = (2 * self.r + 1)**2

        self.fid_conv_proj = nn.Conv2d(fid_dim, self.dim, kernel_size=3, padding=1)
        self.fid_conv_q = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.fid_conv_k = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.fid_conv_v = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.fid_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
        k = 0
        if self.fid:
            k += 1
        if self.perc:
            self.perc_conv_proj = nn.Conv2d(prec_dim, self.dim, kernel_size=3, padding=1)
            self.perc_conv_q = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_conv_k = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_conv_v = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            k += 1
        if self.perc_fid:
            self.perc_fid_conv_q = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_fid_conv_k = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_fid_conv_v = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.perc_fid_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            k += 1
        if self.fid_perc:
            self.fid_perc_conv_q = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.fid_perc_conv_k = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.fid_perc_conv_v = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            self.fid_perc_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            k += 1
        self.k = k
        assert k > 0, 'Need to select a path'

        if self.blend:
            if self.fid:
                self.fid_conv_cond = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                self.fid_cond_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            if self.perc:
                self.perc_conv_cond = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                self.perc_cond_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            if self.perc_fid:
                self.perc_fid_conv_cond = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                self.perc_fid_cond_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)
            if self.fid_perc:
                self.fid_perc_conv_cond = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
                self.fid_perc_cond_proj = nn.Linear(self.dim * self.r_area + 2, self.dim)

            self.cond_norm = nn.LayerNorm(k * self.dim)

            self.blend_proj = nn.Linear(k * self.dim, k)

        self.feat_norm = nn.LayerNorm(self.dim)

    def gen_feat(self, inp_fid_feat, inp_perc_feat, cell):
        if self.fid:
            fid_feat = self.fid_conv_proj(inp_fid_feat)
            self.fid_feat_q = self.fid_conv_q(fid_feat)
            self.fid_feat_k = self.fid_conv_k(fid_feat)
            self.fid_feat_v = self.fid_conv_v(fid_feat)
            self.fid_feat_cond = self.fid_conv_cond(fid_feat) if self.blend else None
        if self.perc:
            perc_feat = self.perc_conv_proj(inp_perc_feat)
            self.perc_feat_q = self.perc_conv_q(perc_feat)
            self.perc_feat_k = self.perc_conv_k(perc_feat)
            self.perc_feat_v = self.perc_conv_v(perc_feat)
            self.perc_feat_cond = self.perc_conv_cond(perc_feat) if self.blend else None
        if self.perc_fid:
            self.perc_fid_feat_q = self.perc_fid_conv_q(perc_feat)
            self.perc_fid_feat_k = self.perc_fid_conv_k(fid_feat)
            self.perc_fid_feat_v = self.perc_fid_conv_v(fid_feat)
            self.perc_fid_feat_cond = self.perc_fid_conv_cond(fid_feat) if self.blend else None
        if self.fid_perc:
            self.fid_perc_feat_q = self.fid_perc_conv_q(fid_feat)
            self.fid_perc_feat_k = self.fid_perc_conv_k(perc_feat)
            self.fid_perc_feat_v = self.fid_perc_conv_v(perc_feat)
            self.fid_perc_feat_cond = self.fid_perc_conv_cond(perc_feat) if self.blend else None


    def query_feat(self, feat_q, feat_k, feat_v, feat_cond, sample_coord, rel_cell):
        bs, q_sample = sample_coord.shape[:2]

        # b, q, 1, 2
        sample_coord_q = sample_coord.clone()
        sample_coord_q = sample_coord_q

        # b, 2, h, w -> b, 2, q, 1 -> b, q, 1, 2
        coord_k = make_coord(feat_k.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
                              unsqueeze(0).expand(bs, 2, *feat_k.shape[-2:])
        sample_coord_k = F.grid_sample(
            coord_k, sample_coord_q.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        # field radius (global: [-1, 1])
        rh = 2 / feat_k.shape[-2]
        rw = 2 / feat_k.shape[-1]
        r = self.r
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(dh, dw, indexing='ij'), axis=-1).view(1, 1, -1, 2)

        # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
        sample_feat_q = F.grid_sample(
            feat_q, sample_coord_q.flip(-1), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)
        sample_feat_q = sample_feat_q.reshape(
            bs, q_sample, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)

        # b, q, 1, 2 -> b, q, r_area, 2
        sample_coord_k = sample_coord_k + delta

        # K - b, c, h, w -> b, c, q, r_area -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, r_area
        sample_feat_k = F.grid_sample(
            feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        sample_feat_k = sample_feat_k.reshape(
            bs, q_sample, self.r_area, self.head, self.dim // self.head
        ).permute(0, 1, 3, 4, 2)

        # V - b, c, h, w -> b, c, q, r_area -> b, q, r_area, c
        sample_feat_v = F.grid_sample(
            feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)

        # b, q, h, 1, r_area -> b, q, r_area, h
        attn = torch.matmul(sample_feat_q, sample_feat_k).reshape(
            bs, q_sample, self.head, self.r_area
        ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

        # b, q, r_area, 2
        rel_coord = sample_coord_q - sample_coord_k
        rel_coord[..., 0] *= feat_k.shape[-2]
        rel_coord[..., 1] *= feat_k.shape[-1]

        _, pb = self.pb_encoder(rel_coord)
        attn = F.softmax(torch.add(attn, pb), dim=-2)

        attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)
        sample_feat_v = sample_feat_v.reshape(
            bs, q_sample, self.r_area, self.head, self.dim // self.head
        )
        sample_feat_v = torch.mul(sample_feat_v, attn).reshape(bs, q_sample, self.r_area, -1)

        feat_out = sample_feat_v.reshape(bs, q_sample, -1)

        feat_out = torch.cat([feat_out, rel_cell], dim=-1)

        if feat_cond is not None:
            sample_feat_cond = F.grid_sample(
                feat_cond, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1)

            sample_feat_cond = sample_feat_cond.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            )
            sample_feat_cond = torch.mul(sample_feat_cond, attn).reshape(bs, q_sample, self.r_area, -1)

            feat_cond = sample_feat_cond.reshape(bs, q_sample, -1)

            feat_cond = torch.cat([feat_cond, rel_cell], dim=-1)

        return feat_out, feat_cond

    def query_rgb(self, sample_coord, cell):
        sample_coord = sample_coord.unsqueeze(2)
        bs, q_sample = sample_coord.shape[:2]

        # b, 2 -> b, q, 2
        rel_cell = cell.clone()
        rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= self.inp.shape[-2]
        rel_cell[..., 1] *= self.inp.shape[-1]

        outs = []
        conds = []
        if self.fid:
            fid_out, fid_cond = self.query_feat(self.fid_feat_q, self.fid_feat_k, self.fid_feat_v, 
                                                self.fid_feat_cond, sample_coord, rel_cell)
            fid_out = self.fid_proj(fid_out)
            outs.append(fid_out)
            if self.blend:
                fid_cond = self.fid_cond_proj(fid_cond)
                conds.append(fid_cond)
        if self.perc:
            perc_out, perc_cond = self.query_feat(self.perc_feat_q, self.perc_feat_k, self.perc_feat_v, 
                                                  self.perc_feat_cond, sample_coord, rel_cell)
            perc_out = self.perc_proj(perc_out)
            outs.append(perc_out)
            if self.blend:
                perc_cond = self.perc_cond_proj(perc_cond)
                conds.append(perc_cond)
        if self.perc_fid:
            perc_fid_out, perc_fid_cond = self.query_feat(self.perc_fid_feat_q, self.perc_fid_feat_k, self.perc_fid_feat_v, 
                                                          self.perc_fid_feat_cond, sample_coord, rel_cell)
            perc_fid_out = self.perc_fid_proj(perc_fid_out)
            outs.append(perc_fid_out)
            if self.blend:
                perc_fid_cond = self.perc_fid_cond_proj(perc_fid_cond)
                conds.append(perc_fid_cond)
        if self.fid_perc:
            fid_perc_out, fid_perc_cond = self.query_feat(self.fid_perc_feat_q, self.fid_perc_feat_k, self.fid_perc_feat_v, 
                                                          self.fid_perc_feat_cond, sample_coord, rel_cell)
            fid_perc_out = self.fid_perc_proj(fid_perc_out)
            outs.append(fid_perc_out)
            if self.blend:
                fid_perc_cond = self.fid_perc_cond_proj(fid_perc_cond)
                conds.append(fid_perc_cond)

        if self.blend:
            cond = torch.cat(conds, dim=-1)
            cond = self.cond_norm(cond)
            cond = nonlinearity(cond)

            blend_weight = self.blend_proj(cond)
            # per-vector
            if self.training:
                blend_weight = F.gumbel_softmax(blend_weight, tau=self.temperature, hard=True, dim=-1).unsqueeze(2)
            else:
                indices = torch.argmax(blend_weight, dim=-1)
                blend_weight = F.one_hot(indices, num_classes=self.k).float().unsqueeze(2)
            blend_out = (blend_weight @ torch.stack(outs, dim=2)).squeeze(2)

        residual = F.grid_sample(self.inp, sample_coord.flip(-1), mode='bilinear',\
                padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        preds = {}
        if self.fid:
            fid_pred = self.imnet_de(self.feat_norm(fid_out))
            preds['fid_pred'] = fid_pred + residual
        if self.perc:
            perc_pred = self.imnet_de(self.feat_norm(perc_out))
            preds['perc_pred'] = perc_pred + residual
        if self.perc_fid:
            perc_fid_pred = self.imnet_de(self.feat_norm(perc_fid_out))
            preds['perc_fid_pred'] = perc_fid_pred + residual
        if self.fid_perc:
            fid_perc_pred = self.imnet_de(self.feat_norm(fid_perc_out))
            preds['fid_perc_pred'] = fid_perc_pred + residual
        if self.blend:
            blend_pred = self.imnet_de(self.feat_norm(blend_out))
            preds['blend_pred'] = blend_pred + residual

        return preds

    def batched_predict(self, coord, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            out = []
            while ql < n:
                qr = min(ql + bsize, n)
                preds = self.query_rgb(coord[:, ql: qr, :], cell)
                if self.blend:
                    pred = preds['blend_pred']
                elif self.fid:
                    pred = preds['fid_pred']
                elif self.perc:
                    pred = preds['perc_pred']
                out.append(pred)
                ql = qr
            out = torch.cat(out, dim=1)
        return out

    def forward(self, inp, inp_fid_feat=None, inp_perc_feat=None, coord=None, cell=None, 
                output_size=None, return_img=False, bsize=65536):
        self.inp = inp
        bs, _, h, w = inp.shape
        device = inp.device

        if coord is None:
            assert output_size is not None
            coord, cell = make_coord_cell(bs, output_size[0], output_size[1])
            coord = coord.to(device)
            cell = cell.to(device)
        if cell is None:
            assert output_size is not None
            cell = torch.ones(2)
            cell[0] *= 2. / output_size[0]
            cell[1] *= 2. / output_size[1]
            cell = cell.unsqueeze(0).repeat(bs, 1)
            cell = cell.to(device)
        if len(coord.shape) == 4:
            coord = coord.reshape(bs, -1, 2)

        self.gen_feat(inp_fid_feat, inp_perc_feat, cell)

        if self.training:
            bsize = 0
        if bsize > 0:
            out = self.batched_predict(coord, cell, bsize)
        else:
            out = self.query_rgb(coord, cell)

        if return_img:
            assert output_size is not None
            out = rearrange(out, 'b (h w) c -> b c h w', h=output_size[0], w=output_size[1])

        return out

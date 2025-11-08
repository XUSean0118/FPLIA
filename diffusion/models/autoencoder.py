import math
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

import models
from diffusion.modules.distributions import DiagonalGaussianDistribution
from diffusion.modules.model import Encoder, Decoder, LIA
from utils import instantiate_from_config, make_coord_cell


def disabled_train(self, mode=True):
    return self

class Autoencoder(pl.LightningModule):
    def __init__(self,
                 aeconfig,
                 fidconfig,
                 liaconfig,
                 sample_q=None,
                 loss_type='l2',
                 scheduler_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 freeze_fid_encoder=True,
                 ):
        super().__init__()
        self.perc_encoder = Encoder(**aeconfig)
        self.perc_decoder = Decoder(**aeconfig)
        self.fid_encoder = models.make(fidconfig).cuda()
        self.lia = LIA(**liaconfig, 
                        fid_dim=self.fid_encoder.out_dim, 
                        prec_dim=self.perc_decoder.out_dim
                    )
        
        self.sample_q = sample_q
        self.loss_type = loss_type
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
    
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if freeze_fid_encoder:
            self.fid_encoder.eval()
            self.fid_encoder.train = disabled_train
            for param in self.fid_encoder.parameters():
                param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' in ckpt:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif 'model' in ckpt:
            sd = torch.load(path, map_location="cpu")["model"]['sd']
        else:
            raise NotImplementedError

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, chop_size=2**20):
        if self.training:
            z = self.perc_encoder(x)
        else:
            z = self.encode_chop(x, chop_size)
        return z

    def encode_chop(self, x, chop_size=2**20):
        # height, width
        h, w = x.shape[-2:]

        if h * w < chop_size:
            z = self.perc_encoder(x)
            return z
        else:
            shave = 16
            down_scale = 2**(self.perc_encoder.num_resolutions - 1)
            top = slice(0, h//2 + (down_scale - h//2 % down_scale) + shave)
            bottom = slice(h - h//2 - (down_scale - (h - h//2) % down_scale) - shave, h)
            left = slice(0, w//2 + (down_scale - w//2 % down_scale) + shave)
            right = slice(w - w//2 - (down_scale - (w - w//2) % down_scale) - shave, w)

            x_chops = [
                x[..., top, left],
                x[..., top, right],
                x[..., bottom, left],
                x[..., bottom, right]
            ]

            z_chops = []
            for i in range(4):
                z_chops.append(self.encode_chop(x_chops[i], chop_size))

            h = h // down_scale
            w = w // down_scale
            top = slice(0, h//2)
            bottom = slice(h - math.ceil(h/2), h)
            bottom_r = slice(h//2 - h, None)
            left = slice(0, w//2)
            right = slice(w - math.ceil(w/2), w)
            right_r = slice(w//2 - w, None)

            # batch size, number of color channels
            b, c = z_chops[0].shape[:2]
            z = torch.zeros(b, c, h, w).to(x.device)

            z[..., top, left] = z_chops[0][..., top, left]
            z[..., top, right] = z_chops[1][..., top, right_r]
            z[..., bottom, left] = z_chops[2][..., bottom_r, left]
            z[..., bottom, right] = z_chops[3][..., bottom_r, right_r]
            del z_chops

            return z

    def decode(self, z, x_lr, coord=None, cell=None, fid_feat=None,
               chop_size=2**20, output_size=None, return_img=False, bsize=36864):
        if self.training:
            perc_feat = self.perc_decoder(z)
        else:
            perc_feat = self.decode_chop(z)
        if self.lia.fid and fid_feat == None:
            fid_feat = self.fid_encoder(x_lr)
        out = self.lia(x_lr,
                       inp_fid_feat=fid_feat,
                       inp_perc_feat=perc_feat, 
                       coord=coord, 
                       cell=cell, 
                       output_size=output_size, 
                       return_img=return_img, 
                       bsize=bsize)
        return out

    def decode_chop(self, z, chop_size=2**20):
        # height, width
        h, w = z.shape[-2:]
        up_scale = 2**(self.perc_decoder.num_resolutions - 1)

        if h * w < chop_size // up_scale // up_scale:
            x = self.perc_decoder(z)
            return x
        else:
            shave = 4
            top = slice(0, h//2 + shave)
            bottom = slice(h - h//2 - shave, h)
            left = slice(0, w//2 + shave)
            right = slice(w - w//2 - shave, w)

            z_chops = [
                z[..., top, left],
                z[..., top, right],
                z[..., bottom, left],
                z[..., bottom, right]
            ]

            x_chops = []
            for i in range(4):
                x_chops.append(self.decode_chop(z_chops[i], chop_size))

            top = slice(0, h//2)
            bottom = slice(h - h//2, h)
            bottom_r = slice(h//2 - h, None)
            left = slice(0, w//2)
            right = slice(w - w//2, w)
            right_r = slice(w//2 - w, None)

            # batch size, number of color channels
            b, c = x_chops[0].shape[:2]
            x = torch.zeros(b, c, h, w).to(z.device)

            x[..., top, left] = x_chops[0][..., top, left]
            x[..., top, right] = x_chops[1][..., top, right_r]
            x[..., bottom, left] = x_chops[2][..., bottom_r, left]
            x[..., bottom, right] = x_chops[3][..., bottom_r, right_r]
            del x_chops

            return x

    def forward(self, x, x_lr, coord=None, cell=None, chop_size=2**20,
                output_size=None, return_img=False, bsize=36864):
        z = self.encode(x, chop_size)
        out = self.decode(z, x_lr, 
                          coord=coord, 
                          cell=cell, 
                          chop_size=chop_size,
                          output_size=output_size, 
                          return_img=return_img, 
                          bsize=bsize)
        return out

    def get_input(self, batch, split='train', sample=True):
        hr = batch['gt']
        hr = hr.to(memory_format=torch.contiguous_format).float()
        lr = batch['inp']
        lr = lr.to(memory_format=torch.contiguous_format).float()

        if split == 'train':
            scale = random.uniform(1, 4) / 4
        else:
            scale = 1.0
        gt = F.interpolate(hr, scale_factor=scale, mode='bicubic')
        bs, _, gt_h, gt_w = gt.shape
        coord, cell = make_coord_cell(bs, gt_h, gt_w)

        if self.sample_q and sample:
            sample_coords = []
            for i in range(bs):
                sample_list = np.random.choice(coord.shape[1], self.sample_q, replace=False)
                sample_coord = coord[i, sample_list, :]
                sample_coords.append(sample_coord)
            coord = torch.stack(sample_coords, dim=0)
        return gt, hr, lr, coord.to(self.device), cell.to(self.device)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = F.mse_loss(target, pred)
            else:
                loss = F.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def training_step(self, batch, batch_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='train')
        preds = self(hr, lr, coord=coord, cell=cell)

        sample_coord = coord.unsqueeze(2)
        gt = F.grid_sample(gt, sample_coord.flip(-1), mode='nearest', align_corners=False)
        gt = gt.squeeze(-1).permute(0, 2, 1)
        
        assert 'perc_pred' in preds
        perc_loss = self.get_loss(gt, preds['perc_pred'])
        self.log("train/perc_loss", perc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        loss = perc_loss

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='val')
        pred = self(hr, lr, coord=coord, cell=cell)

        sample_coord = coord.unsqueeze(2)
        gt = F.grid_sample(gt, sample_coord.flip(-1), mode='nearest', align_corners=False)
        gt = gt.squeeze(-1).permute(0, 2, 1)

        log_dict = {}

        rec_loss = self.get_loss(gt, pred)
        log_dict.update({"val/rec_loss": rec_loss})

        gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [gt, pred]]
        mse = (gt - pred).pow(2).mean()
        psnr = -10 * torch.log10(mse)
        log_dict.update({"val/psnr": psnr})
        
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.perc_encoder.parameters()) + list(self.perc_decoder.parameters())
        params = params + list(set(self.lia.parameters()) - 
                               set(self.lia.pb_encoder.parameters()) -
                               set(self.lia.imnet_de.parameters()))
        opt = torch.optim.Adam(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler_config = OmegaConf.to_container(self.scheduler_config)
            scheduler_config['params'].update({'optimizer': opt})
            scheduler = instantiate_from_config(scheduler_config)

            print("Setting up scheduler...")
            scheduler = [
                {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                },
            ]
            return [opt], scheduler

        return opt
        
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        gt, hr, lr, coord, cell = self.get_input(batch, split='val', sample=False)
        if not only_inputs:
            xrec = self(hr, lr, coord=coord, cell=cell, bsize=36864)
            xrec = rearrange(xrec, 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])
            log["reconstructions"] = xrec
        log["hr"] = hr
        log["lr"] = lr
        return log


class AutoencoderKL(Autoencoder):
    def __init__(self,
                 aeconfig,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 **kwargs,
                 ):

        super().__init__(aeconfig=aeconfig, **kwargs)
        embed_dim = aeconfig["z_channels"]
        self.quant_conv = torch.nn.Conv2d(2*embed_dim, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)

        if ckpt_path is not None:
            super().init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if lossconfig:
            self.loss = instantiate_from_config(lossconfig)

    def encode(self, x, chop_size=2**20):
        if self.training:
            h = self.perc_encoder(x)
        else:
            h = self.encode_chop(x, chop_size)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, x_lr, coord=None, cell=None, fid_feat=None,
               chop_size=2**20, output_size=None, return_img=False, bsize=36864):
        z = self.post_quant_conv(z)
        if self.training:
            perc_feat = self.perc_decoder(z)
        else:
            perc_feat = self.decode_chop(z)
        if self.lia.fid and fid_feat == None:
            fid_feat = self.fid_encoder(x_lr)
        out = self.lia(x_lr,
                       inp_fid_feat=fid_feat,
                       inp_perc_feat=perc_feat, 
                       coord=coord, 
                       cell=cell, 
                       output_size=output_size, 
                       return_img=return_img, 
                       bsize=bsize)
        return out

    def forward(self, x, x_lr, coord=None, cell=None, chop_size=2**20,
                output_size=None, return_img=False, bsize=65536, 
                sample_posterior=True, return_posterior=False):
        posterior = self.encode(x, chop_size)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        out = self.decode(z, x_lr, coord, cell, 
                          chop_size=chop_size,
                          output_size=output_size, 
                          return_img=return_img, 
                          bsize=bsize)
        if return_posterior:
            return out, posterior
        else:
            return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='train')
        preds, posterior = self(hr, lr, coord=coord, cell=cell, return_posterior=True)
        
        assert 'perc_pred' in preds
        pred = rearrange(preds['perc_pred'], 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])

        if optimizer_idx == 0:
            # train encoder+decoder+logvar          
            perc_aeloss, log_dict_ae = self.loss(gt, pred, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train_perc")
            self.log("perc_aeloss", perc_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            loss = perc_aeloss

            self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if optimizer_idx == 1:
            # train the discriminator
            perc_discloss, log_dict_disc = self.loss(gt, pred, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train_perc")
            self.log("perc_discloss", perc_discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            loss = perc_discloss

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='val')
        pred, posterior = self(hr, lr, coord=coord, cell=cell, return_posterior=True)

        pred = rearrange(pred, 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])

        aeloss, log_dict_ae = self.loss(gt, pred, posterior, 0, self.global_step, 
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(gt, pred, posterior, 1, self.global_step, 
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [gt, pred]]
        mse = (gt - pred).pow(2).mean()
        psnr = -10 * torch.log10(mse)
        self.log("val/psnr", psnr)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (list(self.perc_encoder.parameters())+
                  list(self.perc_decoder.parameters())+
                  list(self.quant_conv.parameters())+
                  list(self.post_quant_conv.parameters()))
        params = params + list(set(self.lia.parameters()) - 
                               set(self.lia.pb_encoder.parameters()) -
                               set(self.lia.imnet_de.parameters()))
        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler_config = OmegaConf.to_container(self.scheduler_config)
            scheduler_config['params'].update({'optimizer': opt_ae})
            scheduler = instantiate_from_config(scheduler_config)
            scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=1000, gamma=1)
            print("Setting up scheduler...")
            scheduler = [
                {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                },
                {
                    'scheduler': scheduler_disc,
                    'interval': 'epoch',
                    'frequency': 1
                }
            ]
            return [opt_ae, opt_disc], scheduler

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.lia.imnet_de.layers[-1].weight


class VQModel(Autoencoder):
    def __init__(self,
                 aeconfig,
                 lossconfig,
                 n_embed,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 ckpt_path=None,
                 ignore_keys=[],
                 **kwargs,
                 ):
        super().__init__(aeconfig=aeconfig, **kwargs)
        embed_dim = aeconfig["z_channels"]
        self.quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)

        self.n_embed = n_embed
        self.lr_g_factor = lr_g_factor
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)

        if ckpt_path is not None:
            super().init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)

    def encode(self, x, chop_size=2**20):
        if self.training:
            h = self.perc_encoder(x)
        else:
            h = self.encode_chop(x, chop_size)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, x_lr, coord=None, cell=None, fid_feat=None,
               chop_size=2**20, output_size=None, return_img=False, bsize=36864):
        quant = self.post_quant_conv(quant)
        if self.training:
            perc_feat = self.perc_decoder(quant)
        else:
            perc_feat = self.decode_chop(quant)
        if self.lia.fid and fid_feat == None:
            fid_feat = self.fid_encoder(x_lr)
        out = self.lia(x_lr,
                       inp_fid_feat=fid_feat,
                       inp_perc_feat=perc_feat, 
                       coord=coord, 
                       cell=cell, 
                       output_size=output_size, 
                       return_img=return_img, 
                       bsize=bsize)
        return out

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x, x_lr, coord=None, cell=None, chop_size=2**20,
                output_size=None, return_img=False, bsize=65536, 
                return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(x, chop_size)
        out = self.decode(quant, x_lr, coord, cell, 
                          chop_size=chop_size,
                          output_size=output_size, 
                          return_img=return_img, 
                          bsize=bsize)
        if return_pred_indices:
            return out, diff, ind
        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='train')
        preds, qloss, ind = self(hr, lr, coord=coord, cell=cell, return_pred_indices=True)

        assert 'perc_pred' in preds
        pred = rearrange(preds['perc_pred'], 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])

        if optimizer_idx == 0:
            # autoencode
            perc_aeloss, log_dict_ae = self.loss(qloss, gt, pred, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train_perc")
            self.log("perc_aeloss", perc_aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            loss = perc_aeloss

            self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if optimizer_idx == 1:
            # discriminator
            perc_discloss, log_dict_disc = self.loss(qloss, gt, pred, optimizer_idx, self.global_step,
                                                        last_layer=self.get_last_layer(), split="train_perc")
            self.log("perc_discloss", perc_discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            loss = perc_discloss

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        gt, hr, lr, coord, cell = self.get_input(batch, split='val')
        pred, qloss, ind = self(hr, lr, coord=coord, cell=cell, return_pred_indices=True)

        pred = rearrange(pred, 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])

        aeloss, log_dict_ae = self.loss(qloss, gt, pred, 0, self.global_step, 
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, gt, pred, 1, self.global_step, 
                                            last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [gt, pred]]
        mse = (gt - pred).pow(2).mean()
        psnr = -10 * torch.log10(mse)
        self.log("val/psnr", psnr)

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate

        params = (list(self.perc_encoder.parameters())+
                  list(self.perc_decoder.parameters())+
                  list(self.quantize.parameters())+
                  list(self.quant_conv.parameters())+
                  list(self.post_quant_conv.parameters()))
        params = params + list(set(self.lia.parameters()) - 
                               set(self.lia.pb_encoder.parameters()) -
                               set(self.lia.imnet_de.parameters()))

        opt_ae = torch.optim.Adam(params, lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler_config = OmegaConf.to_container(self.scheduler_config)
            scheduler_config['params'].update({'optimizer': opt_ae})
            scheduler = instantiate_from_config(scheduler_config)
            scheduler_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=1000, gamma=1)
            print("Setting up scheduler...")
            scheduler = [
                {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                },
                {
                    'scheduler': scheduler_disc,
                    'interval': 'epoch',
                    'frequency': 1
                }
            ]
            return [opt_ae, opt_disc], scheduler

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.lia.imnet_de.layers[-1].weight


class VQModelInterface(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(lossconfig=None, *args, **kwargs)

    def encode(self, x, chop_size=2**20):
        if self.training:
            h = self.perc_encoder(x)
        else:
            h = self.encode_chop(x, chop_size)
        h = self.quant_conv(h)
        return h

    def decode(self, h, x_lr, coord=None, cell=None, fid_feat=None,
               chop_size=2**20, output_size=None, return_img=False, bsize=36864,
               force_not_quantize=False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        if self.training:
            perc_feat = self.perc_decoder(quant)
        else:
            perc_feat = self.decode_chop(quant)
        if self.lia.fid and fid_feat == None:
            fid_feat = self.fid_encoder(x_lr)
        out = self.lia(x_lr,
                       inp_fid_feat=fid_feat,
                       inp_perc_feat=perc_feat, 
                       coord=coord, 
                       cell=cell, 
                       output_size=output_size, 
                       return_img=return_img, 
                       bsize=bsize)
        return out

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

import models
from diffusion.modules.model import LIA
from utils import instantiate_from_config, make_coord_cell


def disabled_train(self, mode=True):
    return self

class LIAModel(pl.LightningModule):
    def __init__(self,
                 fidconfig,
                 liaconfig,
                 loss_type='l2',
                 scheduler_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 ):
        super().__init__()
        self.fid_encoder = models.make(fidconfig).cuda()
        self.lia = LIA(**liaconfig, 
                        fid_dim=self.fid_encoder.out_dim, 
                    )
        self.loss_type = loss_type
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
    
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def forward(self, x_lr, coord=None, cell=None, chop_size=2**20,
                output_size=None, return_img=False, bsize=65536):
        fid_feat = self.fid_encoder(x_lr)
        out = self.lia(x_lr,
                       inp_fid_feat=fid_feat, 
                       coord=coord, 
                       cell=cell, 
                       output_size=output_size, 
                       return_img=return_img, 
                       bsize=bsize)
        return out

    def get_input(self, batch, split='train'):
        hr = batch['gt']
        hr = hr.to(memory_format=torch.contiguous_format).float()
        lr = batch['inp']
        lr = lr.to(memory_format=torch.contiguous_format).float()
        if split != 'test':
            coord = batch['coord']
            coord = coord.to(memory_format=torch.contiguous_format).float()
            cell = batch['cell']
            cell = cell.to(memory_format=torch.contiguous_format).float()
            return hr, lr, coord, cell
        else:
            bs, _, gt_h, gt_w = hr.shape
            coord, cell = make_coord_cell(bs, gt_h, gt_w)
            return hr, lr, coord.to(self.device), cell.to(self.device)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def training_step(self, batch, batch_idx):
        gt, lr, coord, cell = self.get_input(batch, split='train')
        pred = self(lr, coord, cell)['fid_pred']

        sample_coord = coord.unsqueeze(2)
        gt = F.grid_sample(gt, sample_coord.flip(-1), mode='nearest', align_corners=False)
        gt = gt.squeeze(-1).permute(0, 2, 1)

        rec_loss = self.get_loss(gt, pred)
        self.log("train/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return rec_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        gt, lr, coord, cell = self.get_input(batch, split='val')
        pred = self(lr, coord, cell)

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
        opt = torch.optim.Adam(self.parameters(), lr=lr)

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
        gt, lr, coord, cell = self.get_input(batch, split='test', sample=False)
        if not only_inputs:
            xrec = self(lr, coord=coord, cell=cell, bsize=36864)
            xrec = rearrange(xrec, 'b (h w) c -> b c h w', h=gt.shape[2], w=gt.shape[3])
            log["reconstructions"] = xrec
        log["hr"] = gt
        log["lr"] = lr
        return log

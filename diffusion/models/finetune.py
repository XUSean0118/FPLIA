"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

import models
from diffusion.models.ddpm import LatentDiffusion
from diffusion.modules.utils import extract_into_tensor
from utils import default, instantiate_from_config, make_coord_cell


class FinetuneModel(LatentDiffusion):
    """main class"""
    def __init__(self, sample_q, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_q = sample_q

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.train()
    
    @torch.no_grad()
    def get_input(self, batch, force_c_encode=False, bs=None, return_first_stage_output=False, 
                  return_original_cond=False, return_x=False, split='train', sample=True):
        hr = super(LatentDiffusion, self).get_input(batch, 'gt')
        lr = super(LatentDiffusion, self).get_input(batch, 'inp')
        if bs is not None:
            hr = hr[:bs]
            lr = lr[:bs]

        hr_h, hr_w = hr.shape[-2:]
        lr_h, lr_w = lr.shape[-2:]

        hr_resize = F.interpolate(hr, size=(lr_h * 4, lr_w * 4), mode='bicubic')
        encoder_posterior = self.encode_first_stage(hr_resize)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
    
        if not self.cond_stage_trainable or force_c_encode:
            c = self.get_learned_conditioning(lr)
        else:
            c = lr
        if bs is not None:
            c = c[:bs]

        bs, _, gt_h, gt_w = hr.shape
        coord, cell = make_coord_cell(bs, gt_h, gt_w)
        coord = coord.to(self.device)
        cell = cell.to(self.device)
            
        if self.sample_q and sample:
            sample_coords = []
            for i in range(bs):
                sample_list = np.random.choice(coord.shape[1], self.sample_q, replace=False)
                sample_coord = coord[i, sample_list, :]
                sample_coords.append(sample_coord)
            sample_coord = torch.stack(sample_coords, dim=0)
            coord = sample_coord

        out = [z, c, coord, cell]

        if return_x:
            out.append(hr)
        if return_original_cond:
            out.append(lr)
        if return_first_stage_output:
            xrec = self.decode_first_stage(z, lr, fid_feat=c,
                                           coord=coord, cell=cell, output_size=(gt_h, gt_w))
            out.append(xrec)
        return out

    def shared_step(self, batch, split='train', **kwargs):
        z, c, sample_coord, cell, gt, lr = self.get_input(batch, split=split, return_x=True, return_original_cond=True)
        loss = self(z, c, sample_coord, cell, gt, lr)
        return loss

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, split='train')

        if self.first_stage_model.lia.temperature >= 0.001 and self.global_step % 1000 == 0:
            self.first_stage_model.lia.temperature *= 0.98

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        _, loss_dict_no_ema = self.shared_step(batch, split='val')
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, split='val')
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def forward(self, z, c, sample_coord, cell, gt, lr, *args, **kwargs):
        t = torch.randint(20, self.num_timesteps-20, (z.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(z, c, t, sample_coord, cell, gt, lr, *args, **kwargs)

    def get_fine_loss(self, pred, target, m, s):
        if self.loss_type == 'l1':
            loss = (m / s) * (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = ((m / s) ** 2) * F.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, sample_coord, cell, gt, lr, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy, mean, std = self.q_sample(x_start=x_start, t=t, noise=noise, return_mean_std=True)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
            x_0_pred = model_output
        elif self.parameterization == "eps":
            target = noise
            x_0_pred = (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_noisy.shape) * x_noisy -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_noisy.shape) * model_output
            )
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        preds = self.decode_first_stage(x_0_pred, lr, coord=sample_coord, cell=cell, #, fid_feat=cond
                                        return_img=False, requires_grad=True)

        sample_coord = sample_coord.unsqueeze(2)
        y = F.grid_sample(gt, sample_coord.flip(-1), mode='nearest', align_corners=False)
        y = y.permute(0, 2, 1, 3)

        if prefix == 'train':
            loss_image = 0
            if 'blend_pred' in preds:
                blend_loss = self.get_fine_loss(y, preds['blend_pred'].unsqueeze(-1), 1, 1).mean()
                self.log("train/blend_loss", blend_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                loss_image += blend_loss
            if 'fid_pred' in preds:
                fid_loss = self.get_fine_loss(y, preds['fid_pred'].unsqueeze(-1), 1, 1).mean()
                self.log("train/fid_loss", fid_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                loss_image += fid_loss
            if 'perc_pred' in preds:
                perc_loss = self.get_fine_loss(y, preds['perc_pred'].unsqueeze(-1), mean, std).mean()
                self.log("train/perc_loss", perc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                loss_image += perc_loss
            if 'perc_fid_pred' in preds:
                perc_fid_loss = self.get_fine_loss(y, preds['perc_fid_pred'].unsqueeze(-1), mean, std).mean()
                self.log("train/perc_fid_loss", perc_fid_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                loss_image += perc_fid_loss
            if 'fid_perc_pred' in preds:
                fid_perc_loss = self.get_fine_loss(y, preds['fid_perc_pred'].unsqueeze(-1), mean, std).mean()
                self.log("train/fid_perc_loss", fid_perc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
                loss_image += fid_perc_loss
            loss_dict.update({'train/loss_image':loss_image})
        else:
            loss_image = self.get_fine_loss(y, preds.unsqueeze(-1), mean, std).mean()
            loss_dict.update({'val/loss_image':loss_image})

        loss = 1.0 * loss_image
        
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=False, plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True, 
                   use_ema_scope=True, **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, coord, cell, gt, lr, xrec = self.get_input(batch,
                                                     split='val',
                                                     sample=False,
                                                     force_c_encode=True,
                                                     return_x=True,
                                                     return_original_cond=True,
                                                     return_first_stage_output=True,
                                                     bs=N)
        output_size = gt.shape[-2:]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["inputs"] = gt
        log["reconstruction"] = xrec

        if self.model.conditioning_key is not None:
            log["conditioning"] = lr

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(z_noisy)

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = self._get_denoise_row_from_list(diffusion_row, lr[:n_row], 
                                                             coord=coord[:n_row], cell=cell[:n_row], 
                                                             fid_feat=c[:n_row], output_size=output_size)
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples, lr[:N], fid_feat=c[:N], 
                                                coord=coord[:N], cell=cell[:N], output_size=output_size)
            log["samples"] = x_samples
            if plot_denoise_rows:
                if use_ddim:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row['x_inter'], lr[:N], 
                                                                   coord=coord[:N], cell=cell[:N],
                                                                   fid_feat=c[:N], output_size=output_size)
                    log["denoise_row_x_inter"] = denoise_grid

                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row['pred_x0'], lr[:N], 
                                                                   coord=coord[:N], cell=cell[:N],
                                                                   fid_feat=c[:N], output_size=output_size)
                    log["denoise_row_pred_x0"] = denoise_grid                    
                else:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row, lr[:N], 
                                                                   coord=coord[:N], cell=cell[:N],
                                                                   fid_feat=c[:N], output_size=output_size)
                    log["denoise_row"] = denoise_grid

            if quantize_denoised:
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device), xc, fid_feat=c)
                log["samples_x0_quantized"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size[0], self.image_size[1]),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, lr[:N], 
                                                       coord=coord[:N], cell=cell[:N],
                                                       fid_feat=c[:N], output_size=output_size,
                                                       desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate

        params = list(self.first_stage_model.lia.parameters())

        opt = torch.optim.AdamW(params, lr=lr)

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

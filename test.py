import argparse
import math
import os
import time
from functools import partial

import lpips
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusion.models.ddim import DDIMSampler
from utils import Averager, calc_psnr, calc_ssim, instantiate_from_config

import datasets

def sample_chop(model, cond, batch_size, ddim_steps, eta, min_size=160*160):
    # height, width
    h, w = cond.shape[-2:]
    
    if h * w < min_size:
        samples, _ = model.sample_log(cond=cond, batch_size=batch_size, shape=(model.channels, h, w),
                                        ddim=True, ddim_steps=ddim_steps, eta=eta, log_every_t=100)
        return samples
    else:
        shave = 32
        down_scale = 8
        top = slice(0, h//2 + (down_scale - h//2 % down_scale) + shave)
        bottom = slice(h - h//2 - (down_scale - (h - h//2) % down_scale) - shave, h)
        left = slice(0, w//2 + (down_scale - w//2 % down_scale) + shave)
        right = slice(w - w//2 - (down_scale - (w - w//2) % down_scale) - shave, w)

        x_chops = [
            cond[..., top, left],
            cond[..., top, right],
            cond[..., bottom, left],
            cond[..., bottom, right]
        ]

        y_chops = []
        for i in range(4):
            y_chops.append(sample_chop(model=model, cond=x_chops[i], batch_size=batch_size, 
                                        ddim_steps=ddim_steps, eta=eta, min_size=min_size))

        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0].shape[:2]
        y = torch.zeros(b, c, h, w).to(cond.device)

        y[..., top, left] = y_chops[0][..., top, left]
        y[..., top, right] = y_chops[1][..., top, right_r]
        y[..., bottom, left] = y_chops[2][..., bottom_r, left]
        y[..., bottom, right] = y_chops[3][..., bottom_r, right_r]
        del y_chops

        return y


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    model.cuda()
    model.eval()
    return model


def evaluation(model_config, dataset_config, ckpt_path, eta=0.0, steps=200, save_image=False):
    model_config = OmegaConf.load(model_config)
    ignore_keys = model_config.model.params.get('ignore_keys', [])
    ignore_keys.append('loss')
    model_config.model.params.ignore_keys = ignore_keys
    model_config.model.params.ckpt_path = ckpt_path

    model = load_model_from_config(model_config, ckpt_path)

    save_path = os.path.join(exp, dataset_config.split('/')[-1].split('.yaml')[0])
    os.makedirs(save_path, exist_ok=True)

    dataset_config = OmegaConf.load(dataset_config)
    spec = OmegaConf.to_container(dataset_config['test_dataset'])
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        num_workers=8,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    eval_type=dataset_config.get('eval_type')
    eval_bsize=dataset_config.get('eval_bsize')

    if eval_type is None:
        psnr_fn = calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    lpips_fn = lpips.LPIPS(net='alex').eval()

    psnr_res = Averager()
    ssim_res = Averager()
    lpips_res = Averager()
    val_time = Averager()
    if model.first_stage_model.lia.blend:
        index = {'fid_index': 0, 'perc_index': 0, 'perc_fid_index': 0, 'fid_perc_index': 0}

    IDX = 0
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        hr = batch['gt']
        lr = batch['inp']
        bs, _, hr_h, hr_w = hr.shape
        lr_h, lr_w = lr.shape[-2:]
        
        coord = batch['coord']
        coord_h, coord_w = coord.shape[1:3]
        cell = batch['cell']

        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            fid_feat = model.get_learned_conditioning(lr)
            with model.ema_scope():
                samples = sample_chop(model=model, cond=fid_feat, batch_size=bs, ddim_steps=steps, eta=eta)
            pred = model.decode_first_stage(samples, lr, coord=coord, cell=cell, fid_feat=fid_feat,
                                            bsize=eval_bsize, return_img=False)

        end = time.time()
        torch.cuda.synchronize()

        val_time.add(end - start, bs)

        gt = hr * 0.5 + 0.5
        final_pred = pred * 0.5 + 0.5
        final_pred.clamp_(0, 1)
        final_pred = final_pred.view(bs, coord_h, coord_w, 3).permute(0, 3, 1, 2)
        final_pred = final_pred[..., :hr_h, :hr_w]

        psnr = psnr_fn(final_pred, gt)
        psnr_res.add(psnr.item(), bs)
        ssim = calc_ssim(final_pred, gt)
        ssim_res.add(ssim.item(), bs)

        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        pred_n = norm(final_pred).detach().cpu()
        gt_n = norm(gt).detach().cpu()

        loss_lpips = lpips_fn(pred_n, gt_n).mean()
        lpips_res.add(loss_lpips.item(), bs)

        IDX += 1
        if save_image:
            img_pred = (final_pred.squeeze().permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
            Image.fromarray(img_pred).save(f'{save_path}/{IDX:03d}_pred.png')

        torch.cuda.empty_cache()

        with open(f'{save_path}/result.txt', mode='a') as f:
            print(f'{IDX:03d} time: {end - start:.6f}', file=f)
            print(f'PSNR result: {psnr.item():.6f}', file=f)
            print(f'SSIM result: {ssim.item():.6f}', file=f)
            print(f'LPIPS result: {loss_lpips.item():.6f}', file=f)
                    
        pbar.set_description(f'pnsr: {psnr_res.item():.4f}, ssim: {ssim_res.item():.4f}, lpips: {lpips_res.item():.4f}')

    with open(f'{save_path}/result.txt', mode='a') as f:
        print(f'PSNR AVG-result: {psnr_res.item():.6f}', file=f)
        print(f'SSIM AVG-result: {ssim_res.item():.6f}', file=f)
        print(f'LPIPS AVG-result: {lpips_res.item():.6f}', file=f)
        print(f'AVG-Time: {val_time.item():.6f}', file=f)

    fin_res = {'PSNR': psnr_res.item(), 'SSIM': ssim_res.item(), 'LPIPS': lpips_res.item()}

    return fin_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Default
    parser.add_argument('--exp', type=str, required=True, help='Path to the exp')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset config')
    parser.add_argument('--steps', type=int, default=200, help='DDIM steps')
    parser.add_argument('--eta', type=float, default=1.0, help='eta of DDIM')
    parser.add_argument('--save_image', action='store_true', default=False, help='Save outputs')

    args = parser.parse_args()

    seed_everything(2454)

    exp = args.exp
    exp_data = exp.split('/')[-1].split('_')[0]
    config_path = os.path.join(exp, 'configs', f'{exp_data}-project.yaml')
    ckpt_path = os.path.join(exp, 'checkpoints', 'last.ckpt')

    fin_res = evaluation(model_config=config_path, dataset_config=args.dataset, ckpt_path=ckpt_path, 
                        eta=args.eta, steps=args.steps, save_image=args.save_image)
    print(fin_res)


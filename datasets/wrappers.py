import random
import math
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

        
@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        sample_q=None,
        scale_align=8,
        augment=False
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.sample_q = sample_q
        self.scale_align = scale_align
        self.augment = augment

    def collate_fn(self, datas):
        batch_size = len(datas)
        scale = datas[0]['img_hr'].shape[-2] // datas[0]['img_lr'].shape[-2]

        hr_list = []
        lr_list = []

        if self.inp_size is None:
            # batch_size: 1
            lr_h = datas[0]['img_lr'].shape[-2]
            lr_w = datas[0]['img_lr'].shape[-1]
            hr_h = lr_h * scale
            hr_w = lr_w * scale
            lr_list.append(datas[0]['img_lr'])
            hr_list.append(datas[0]['img_hr'][..., :hr_h, :hr_w])
        else:
            hr_h = hr_w = self.inp_size * scale
            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['img_lr'].shape[-2] - self.inp_size)
                w0 = random.randint(0, data['img_lr'].shape[-1] - self.inp_size)
                crop_lr = data['img_lr'][:, h0:h0 + self.inp_size, w0:w0 + self.inp_size]
                h1 = h0 * scale
                w1 = w0 * scale
                crop_hr = data['img_hr'][:, h1:h1 + hr_h, w1:w1 + hr_w]
                lr_list.append(crop_lr)
                hr_list.append(crop_hr)

        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
        inp = inp * 2.0 - 1.0
        hr_rgb = hr_rgb * 2.0 - 1.0

        if self.inp_size is None and self.scale_align != 0:
            # SwinIR and UNet Diffusion Evaluation - reflection padding
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.scale_align + 1) * self.scale_align - h_old
            w_pad = (w_old // self.scale_align + 1) * self.scale_align - w_old
            inp = F.pad(inp, (0, w_pad, 0, h_pad), 'reflect')

            lr_h += h_pad
            lr_w += w_pad

        img_h = lr_h * scale
        img_w = lr_w * scale

        coord = make_coord((img_h, img_w),
                            flatten=False).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        cell = torch.ones(2)
        cell[0] *= 2. / img_h
        cell[1] *= 2. / img_w
        cell = cell.unsqueeze(0).repeat(batch_size, 1)

        if self.sample_q is None:
            sample_coord = coord
        else:
            sample_coord = []
            for i in range(len(hr_list)):
                flatten_coord = coord[i].reshape(-1, 2)
                sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                sample_flatten_coord = flatten_coord[sample_list, :]
                sample_coord.append(sample_flatten_coord)
            sample_coord = torch.stack(sample_coord, dim=0)

        return {'inp': inp, 'gt': hr_rgb, 'coord': sample_coord, 'cell': cell}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        return {'img_lr': img_lr, 'img_hr': img_hr}


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    def __init__(
        self,
        dataset,
        inp_size=None,
        scale_min=4,
        scale_max=4,
        sample_q=None,
        k=1,
        scale_align=8,
        augment=False,
        phase='train',
        only_img=False
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.k = k
        self.scale_align = scale_align
        self.augment = augment
        self.phase = phase
        self.only_img = only_img

        self.counter = 0

    def collate_fn(self, datas):
        batch_size = len(datas)

        hr_list = []

        if self.inp_size is None: # test
            # batch_size: 1
            scale = self.scale_max
            lr_h = math.floor(datas[0]['inp'].shape[-2] / scale + 1e-9)
            lr_w = math.floor(datas[0]['inp'].shape[-1] / scale + 1e-9)
            hr_h = round(lr_h * scale)
            hr_w = round(lr_w * scale)
            crop_hr = datas[0]['inp'][:, :hr_h, :hr_w]
            hr_list.append(crop_hr)
        else: # train
            if self.phase == 'train':
                if self.counter % self.k == 0:
                    self.counter = 1
                else:
                    self.counter += 1
                scale = random.uniform(self.scale_min[self.counter - 1], self.scale_max[self.counter - 1])
            else:
                scale = self.scale_max[-1]

            img_size_min = 9999
            for idx, data in enumerate(datas):
                img_size_min = min(img_size_min, data['inp'].shape[-2], data['inp'].shape[-1])
            if scale * self.inp_size > img_size_min:
                img_ratio = img_size_min / (scale * self.inp_size)
                scale *= img_ratio

            lr_h = lr_w = self.inp_size
            hr_h = hr_w = round(self.inp_size * scale)
            for idx, data in enumerate(datas):
                h0 = random.randint(0, data['inp'].shape[-2] - hr_h)
                w0 = random.randint(0, data['inp'].shape[-1] - hr_w)
                crop_hr = data['inp'][:, h0:h0 + hr_h, w0:w0 + hr_w]
                hr_list.append(crop_hr)

        lr_list = [resize_fn(hr_list[i], (lr_h, lr_w)) for i in range(len(hr_list))]
        inp = torch.stack(lr_list, dim=0)
        hr_rgb = torch.stack(hr_list, dim=0)
        inp = inp * 2.0 - 1.0
        hr_rgb = hr_rgb * 2.0 - 1.0
        
        if self.inp_size is None and self.scale_align != 0:
            # SwinIR and UNet Diffusion Evaluation - reflection padding
            h_old, w_old = inp.shape[-2:]
            h_pad = (h_old // self.scale_align + 1) * self.scale_align - h_old
            w_pad = (w_old // self.scale_align + 1) * self.scale_align - w_old
            inp = F.pad(inp, (0, w_pad, 0, h_pad), 'reflect')

            lr_h += h_pad
            lr_w += w_pad

        img_h = round(lr_h * scale)
        img_w = round(lr_w * scale)

        if self.only_img:
            return {'inp': inp, 'gt': hr_rgb}    
        else:
            coord = make_coord((img_h, img_w),
                                flatten=False).unsqueeze(0).repeat(batch_size, 1, 1, 1)

            cell = torch.ones(2)
            cell[0] *= 2. / img_h
            cell[1] *= 2. / img_w
            cell = cell.unsqueeze(0).repeat(batch_size, 1)

            if self.sample_q is None:
                sample_coord = coord
            else:
                sample_coord = []
                for i in range(len(hr_list)):
                    flatten_coord = coord[i].reshape(-1, 2)
                    sample_list = np.random.choice(flatten_coord.shape[0], self.sample_q, replace=False)
                    sample_flatten_coord = flatten_coord[sample_list, :]
                    sample_coord.append(sample_flatten_coord)
                sample_coord = torch.stack(sample_coord, dim=0)

            return {'inp': inp, 'gt': hr_rgb, 'coord': sample_coord, 'cell': cell}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        return {'inp': img}

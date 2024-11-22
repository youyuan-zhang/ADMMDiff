import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from functions.denoising_clip import clip_ddim_diffusion_admm
from functions.denoising_parse import parse_ddim_diffusion_admm
from functions.denoising_sketch import sketch_ddim_diffusion_admm
import torchvision.utils as tvu


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = "fixedsmall"
        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, mode):
        cls_fn = None
        model_f = None
        model_i = None

        if self.args.model_type == "face":
            # get face model
            celeba_dict = {
                'type': "simple",
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': [1, 1, 2, 2, 4, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16, ],
                'dropout': 0.0,
                'var_type': 'fixedsmall',
                'ema_rate': 0.999,
                'ema': True,
                'resamp_with_conv': True,
                "image_size": 256, 
                "resamp_with_conv": True,
                "num_diffusion_timesteps": 1000,
            }
            model_f = Model(celeba_dict)
            ckpt = os.path.join("ckpts/celeba_hq.ckpt")
            states = torch.load(ckpt, map_location=self.device)
            if type(states) == list:
                states_old = states[0]
                states = dict()
                for k, v in states.items():
                    states[k[7:]] = v
            else:
                model_f.load_state_dict(states)
            model_f.to(self.device)
            model_f = torch.nn.DataParallel(model_f)
            model = model_f

        self.sample_sequence(model, cls_fn, mode)

    def sample_sequence(self, model, cls_fn, mode):
        args = self.args
        pbar = tqdm.tqdm(range(1, self.args.batch_size+1))

        for index in pbar:

            x = torch.randn(
                1,
                3,
                256,
                256,
                device=self.device,
            )
            y = None

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            if mode == "clip_ddim_admm":
                x, _ = self.sample_image_alogrithm_clip_ddim_admm(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, prompt=args.prompt, stop=args.stop, domain=args.model_type)
            elif mode == "parse_ddim_admm":
                x, _ = self.sample_image_alogrithm_parse_ddim_admm(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)
            elif mode == "sketch_ddim_admm":
                x, _ = self.sample_image_alogrithm_sketch_ddim_admm(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, stop=args.stop, ref_path=args.ref_path)

            x = [((y + 1.0) / 2.0).clamp(0.0, 1.0) for y in x]

            # for i in [-1]:  # range(len(x)):
            for i in range(len(x)):
                for j in range(x[i].size(0)):
                    tvu.save_image(
                        x[i][j], os.path.join(self.args.image_folder, "output.png")
                    )
    
    def sample_image_alogrithm_clip_ddim_admm(self, x, model, last=True, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = clip_ddim_diffusion_admm(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, prompt=prompt, stop=stop, domain=domain)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_parse_ddim_admm(self, x, model, last=True, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = parse_ddim_diffusion_admm(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
    
    def sample_image_alogrithm_sketch_ddim_admm(self, x, model, last=True, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        x.requires_grad = True
        
        x = sketch_ddim_diffusion_admm(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, stop=stop, ref_path=ref_path)

        if last:
            x = x[0][-1]
        return x
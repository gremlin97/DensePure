import argparse
import os
import torch
import math
import random
import numpy as np
from cm.karras_diffusion import karras_sample
from cm.random_util import get_generator
from cm.karras_diffusion import stochastic_iterative_sampler
from cm.random_util import get_generator
import torch as th
import torch.distributed as dist
from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class CM(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        print("Loading model")
        defaults = model_and_diffusion_defaults()
        model, diffusion = create_model_and_diffusion(**defaults)
        model.load_state_dict(
            dist_util.load_state_dict("pretrained/ct_imagenet64.pt", map_location="cpu")
        )
        model.to(self.device)
        model.convert_to_fp16()
        model.eval()

        self.model = model
        self.diffusion = diffusion
        sigma = self.args.sigma
        a = 1/(1+(sigma*2)**2)
        self.scale = a**0.5
        sigma = sigma*2
        self.s = sigma
        T = self.args.t_total
        self.t = T*(1-(2*1.008*math.asin(math.sin(math.pi/(2*1.008))/(1+sigma**2)**0.5))/math.pi)
        
    def denoiser(self, x_t, s):
        class_cond = True
        model_kwargs = {}
        if class_cond:
            classes = torch.randint(
                low=0,
                high=NUM_CLASSES,
                size=(16,),
                device=dist_util.dev(),
            )
            model_kwargs["y"] = classes
        # x_t = x_t.cpu().detach().numpy()
        # print("Type",type(x_t))
        # print("Dim=============",x_t.ndim)
        _, denoised = self.diffusion.denoise(self.model, x_t, s, **model_kwargs)
        denoised = denoised.clamp(-1, 1)
        return denoised


    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            x0 = img

            x0 = self.scale*(img)
            print(x0.shape)
            print("Dim=============",x0.ndim)
            t = self.t

            if self.args.use_clustering:
                x0 = x0.unsqueeze(1).repeat(1,self.args.clustering_batch,1,1,1).view(batch_size*self.args.clustering_batch,3,32,32)
                
            # Intially we are only experimenting with one step denoising
            if self.args.use_one_step:
                # one step denoise
                t = torch.tensor([round(t)] * x0.shape[0], device=self.device)
                generator = get_generator("determ",1,0)
                
                x_T = x0
                sample = stochastic_iterative_sampler(
                    self.denoiser,
                    x_T,
                    sigmas = sigma,
                    ts = [0,22,39],
                    t_min=0.02,
                    t_max=80,
                    rho=self.diffusion.rho,
                    steps=40,
                    generator=generator,
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                
            print("Output Sample is",sample)
            print("Output Sample shape is",sample.shape)
            return sample
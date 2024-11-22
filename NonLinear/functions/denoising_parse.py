import torch
from tqdm import tqdm
import numpy as np

from .face_parsing.model import FaceParseTool
from .denoising import compute_alpha
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image


def parse_ddim_diffusion_admm(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    parser = FaceParseTool(ref_path=ref_path).cuda()

    ref_img = parser.net(parser.ref)[0][0].detach().cpu().numpy()
    ref_img = np.argmax(ref_img, axis=0) - 1
    ref_img[ref_img == -1] = 18
    cmap = plt.cm.get_cmap('tab20c', 19)
    norm = mcolors.BoundaryNorm(np.arange(0, 19), cmap.N)
    plt.imshow(ref_img, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.savefig("output/seg_guidance.png", bbox_inches='tight', pad_inches=0)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    
    z0 = torch.randn(1, 3, 256, 256).to(x.device)
    zt = z0.clone()
    xt = z0.clone()
    # params
    nu = torch.zeros_like(xt) # dual variable
    max_iter, c = 20, 3 # steps SGD

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            atbar = compute_alpha(b, t.long())[0, 0, 0, 0]
            atbar_prev = compute_alpha(b, next_t.long())[0, 0, 0, 0]

            at = atbar / atbar_prev
            beta_tilde = (1 - at) * (1 - atbar_prev) / (1 - atbar) 
            bt = 1 - at
            if i < 1:
                atbar_prev = torch.tensor(1, device="cuda")
                beta_tilde = torch.tensor(0, device="cuda")
                at = atbar / atbar_prev
                bt = 1 - at
        
        rho = c
        xt = zt - nu / rho
        
        with torch.enable_grad():
            xt = xt.detach().requires_grad_(True)
            et = model(xt, t)
            if et.size(1) == 6:
                et = et[:, :3]
            s = - 1 / torch.sqrt(1 - atbar) * et
        
            pred_x0 = (xt + (1 - atbar) * s) / torch.sqrt(atbar)
        
            residual = parser.get_residual(pred_x0)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            eta = 0.5
            c1 = (1 - atbar_prev).sqrt() * eta
            c2 = (1 - atbar_prev).sqrt() * ((1 - eta ** 2) ** 0.5)
            pred_xt = atbar_prev.sqrt() * pred_x0 + c1 * torch.randn_like(pred_x0) + c2 * et

            # use guided gradient
            cc = atbar.sqrt() * rho_scale
            if i > 800:
                pred_xt -= cc * norm_grad

        pred_xt = pred_xt.detach()
        xt = xt.detach()
        xt = pred_xt

        if i < 100:
            zt = xt
            nu = 0
            continue
        
        for k in range(max_iter):
            with torch.enable_grad():
                zt = zt.detach().requires_grad_()
                pred_z0 = (zt + (1 - atbar) * s) / torch.sqrt(atbar)
                zt = zt + 0.2 * (rho * (xt - zt) + nu)
                if i < 800 and i > 300:
                    zt = zt - 0.3 * cc * norm_grad
                elif i <= 300 and i > 100:
                    zt = zt - 0.1 * cc * norm_grad
            zt = zt.detach()
            
        nu = nu + rho * (xt.data - zt.data)

    # return x0_preds, xs
    return [xt.detach()], None

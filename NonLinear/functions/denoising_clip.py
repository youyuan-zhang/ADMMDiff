import torch
from tqdm import tqdm

from .clip.base_clip import CLIPEncoder
from .denoising import compute_alpha


def clip_ddim_diffusion_admm(x, seq, model, b, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])

    z0 = torch.randn(1, 3, 256, 256).to(x.device)
    zt = z0.clone()
    xt = z0.clone()
    # params
    nu = torch.zeros_like(xt) # dual variable
    max_iter, c = 20, 1 # steps SGD

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
            pred_xt = (xt + (1 - at) * s) / torch.sqrt(at)
        
            residual = clip_encoder.get_residual(pred_x0, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = atbar_prev.sqrt() * (1 - atbar / atbar_prev) / (1 - atbar)
            c2 = (atbar / atbar_prev).sqrt() * (1 - atbar_prev) / (1 - atbar)
            c3 = (1 - atbar_prev) * (1 - atbar / atbar_prev) / (1 - atbar)
            c3 = (c3.log() * 0.5).exp()
            
            l1 = ((et * et).mean().sqrt() * (1 - atbar).sqrt() / atbar.sqrt() * c1).item()
            l2 = l1 * 0.02
            cc = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            if i > 800:
                pred_xt -= 0.5 * cc * norm_grad

        pred_xt = pred_xt.detach()
        xt = xt.detach()

        noise = torch.randn_like(xt).detach()   
        xt = pred_xt + torch.sqrt(beta_tilde) * noise

        if i < 100:
            zt = xt
            nu = 0
            continue
        
        for k in range(max_iter):
            with torch.enable_grad():
                zt = zt.detach().requires_grad_()
                pred_z0 = (zt + (1 - atbar) * s) / torch.sqrt(atbar)
                zt = zt + 0.2 * (rho * (pred_xt - zt) + nu)
                if i < 800 and i > 100:
                    zt = zt - 1 * cc * norm_grad
            zt = zt.detach()
            print(f"timestep: {t}, xtzt diff: {((pred_xt-zt)**2).sum().item()}")
        
        zt = zt + torch.sqrt(beta_tilde) * noise
        nu = nu + rho * (xt.data - zt.data)

    # return x0_preds, xs
    return [xt.detach()], None

import torch
from util.helper import ifft2c, psnr                     # move fft2c / ifft2c into data/data.py
from util.helper import rss
import matplotlib.pyplot as plt
import numpy as np

def ambient_flow_loss(model, Ay, A, delta=0.1, eps=1e-3, center_frac=0.08, gen=None, plot=False):
    """Loss ‖A u_θ(x_t, Ã, t) − u‖² for one batch. gen: torch.Generator for fixed randomness (validation)."""
    device = next(model.parameters()).device
    Ay, A = Ay.to(device, non_blocking=True), A.to(device, non_blocking=True)
    bs, C, H, W = Ay.shape
    Ay = Ay / ifft2c(Ay).abs().amax(dim=(1, 2, 3), keepdim=True)
    A_orig = A.clone()
    n_c = round(W * center_frac); c0, c1 = W // 2 - n_c // 2, W // 2 + n_c // 2
    Bm = (torch.rand(bs, 1, 1, W, device=device, generator=gen) > delta).float()
    Bm[..., c0:c1] = 1
    A_tilde = Bm * A
    t = torch.rand(bs, device=device, generator=gen) * (1 - 2 * eps) + eps
    x0 = torch.complex(torch.randn(Ay.shape, device=device, generator=gen),
                       torch.randn(Ay.shape, device=device, generator=gen)) / 2**0.5

    tt = t.view(-1, 1, 1, 1)
    xt = tt * Ay + (1 - tt) * x0
    u = Ay - x0

    # --- forward pass ---
    inp = torch.cat([xt.real, xt.imag, A_tilde.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])

    # --- plots ---
    if plot:
        show_k = lambda k: torch.log(rss(k[:1])[0].detach().float().cpu() + 1e-6)   # complex → log RSS magnitude
        show_m = lambda m: m[0, 0].expand(H, W).float().cpu()                       # (1, W) column mask → (H, W)

        panels = {"Ay": show_k(Ay), "A (mask)": show_m(A),
                f"u (t={t[0].item():.2f})": show_k(u), "v (prediction)": show_k(v), "A_orig": show_m(A_orig)}
        vmin = min(panels["v (prediction)"].min(), panels[f"u (t={t[0].item():.2f})"].min())
        vmax = max(panels["v (prediction)"].max(), panels[f"u (t={t[0].item():.2f})"].max())
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, (name, img) in zip(ax, panels.items()):
            same = name.startswith(("u", "v"))                                    # u and v share a color scale
            a.imshow(img, cmap="gray", vmin=vmin if same else None, vmax=vmax if same else None)
            a.set_title(name); a.axis("off")
        plt.tight_layout(); plt.show()

    loss = (A * v - u).abs().pow(2).sum() / (A.sum() * C * H)
    baseline = u.abs().pow(2).sum() / (A.sum() * C * H)
    print(f"loss = {loss.item():.4f}   baseline = {baseline.item():.4f}   ratio = {(loss / baseline).item():.3f}")
    return loss


def flow_loss(model, Ay, A, t, eps=1e-3, gen=None, plot=False):
    """Plain flow matching on measured k-space: ‖A(u_θ(x_t, t) − u)‖²."""
    device = next(model.parameters()).device
    Ay, A = Ay.to(device, non_blocking=True), A.to(device, non_blocking=True)
    bs, C, H, W = Ay.shape
    # A_orig = Ay.clone()
    Ay = Ay / Ay.abs().amax(dim=(1, 2, 3), keepdim=True)
    # print(f"Ay.sum() = {Ay.sum().item()}, A_orig.sum() = {A_orig.sum().item()}")
    y0 = torch.complex(torch.randn(Ay.shape, device=device, generator=gen),
                       torch.randn(Ay.shape, device=device, generator=gen)) / 2**0.5

    xt = t * Ay + (1 - t) * y0          # line 3: straight path
    u = Ay - y0                           # line 4: target velocity

    # --- forward pass ---
    inp = torch.cat([xt.real, xt.imag], dim=1)              # 2C channels, no mask channel
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t.view(-1) * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])

    # --- plots ---
    if plot:
        show_k = lambda k: torch.log(rss(k[:1])[0].detach().float().cpu() + 1e-6)   # complex → log RSS magnitude
        show_m = lambda m: m[0, 0].expand(H, W).float().cpu()                       # (1, W) column mask → (H, W)

        t0 = t[0].item()
        key_u, key_v = f"u (t={t0:.2f})", "v (prediction)"
        panels = {
            "Ay": show_k(Ay),
            f"xt (t={t0:.2f}) | psnr = {psnr(xt, Ay):.2f}": show_k(xt),
            "A (mask)": show_m(A),
            key_u: show_k(u),
            key_v: show_k(v),
        }
        vmin = min(panels["v (prediction)"].min(), panels[f"u (t={t[0].item():.2f})"].min())
        vmax = max(panels["v (prediction)"].max(), panels[f"u (t={t[0].item():.2f})"].max())

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for a, (name, img) in zip(ax, panels.items()):
            same = name.startswith(("u", "v"))                                    # u and v share a color scale
            a.imshow(img, cmap="gray", vmin=vmin if same else None, vmax=vmax if same else None)
            a.set_title(name); a.axis("off")
        plt.tight_layout(); plt.show()

    N = A.sum().item()
    loss = (A * (v - u)).abs().pow(2).sum()            # line 5, on measured columns only
    baseline = (A * u).abs().pow(2).sum()

    return loss / N, baseline / N
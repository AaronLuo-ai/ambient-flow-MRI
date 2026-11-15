
import torch
from util.helper import ifft2c                     # move fft2c / ifft2c into data/data.py
from util.helper import rss
import matplotlib.pyplot as plt
import numpy as np

def ambient_loss(model, y, A, delta=0.1, eps=1e-3, center_frac=0.08, gen=None):
    """Loss ‖A u_θ(x_t, Ã, t) − u‖² for one batch. gen: torch.Generator for fixed randomness (validation)."""
    device = next(model.parameters()).device
    y, A = y.to(device, non_blocking=True), A.to(device, non_blocking=True)
    bs, C, H, W = y.shape
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)

    n_c = round(W * center_frac); c0, c1 = W // 2 - n_c // 2, W // 2 + n_c // 2
    Bm = (torch.rand(bs, 1, 1, W, device=device, generator=gen) > delta).float()
    Bm[..., c0:c1] = 1
    A_tilde = Bm * A
    t = torch.rand(bs, device=device, generator=gen) * (1 - 2 * eps) + eps
    x0 = torch.complex(torch.randn(y.shape, device=device, generator=gen),
                       torch.randn(y.shape, device=device, generator=gen)) / 2**0.5

    tt = t.view(-1, 1, 1, 1)
    xt = tt * y + (1 - tt) * (Bm * y + x0)
    u = y - (Bm * y + x0)

    inp = torch.cat([xt.real, xt.imag, A_tilde.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])
    return (A * v - u).abs().pow(2).sum() / (A.sum() * C * H)

def ambient_loss_with_plot(model, y, A, delta=0.1, eps=1e-3, center_frac=0.08, gen=None):
    """Loss ‖A u_θ(x_t, Ã, t) − u‖² for one batch. gen: torch.Generator for fixed randomness (validation)."""
    device = next(model.parameters()).device
    y, A = y.to(device, non_blocking=True), A.to(device, non_blocking=True)
    bs, C, H, W = y.shape
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)

    n_c = round(W * center_frac); c0, c1 = W // 2 - round(n_c) // 2, W // 2 + round(n_c) // 2
    Bm = (torch.rand(bs, 1, 1, W, device=device, generator=gen) > delta).float()
    Bm[..., c0:c1] = 1
    A_tilde = Bm * A
    t = torch.rand(bs, device=device, generator=gen) * (1 - 2 * eps) + eps
    x0 = torch.complex(torch.randn(y.shape, device=device, generator=gen),
                       torch.randn(y.shape, device=device, generator=gen)) / 2**0.5

    tt = t.view(-1, 1, 1, 1)
    xt = tt * y + (1 - tt) * (Bm * y + x0)
    u = y - (Bm * y + x0)

    # --- forward pass ---
    inp = torch.cat([xt.real, xt.imag, A_tilde.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])

    # --- plots ---
    show_k = lambda k: torch.log(rss(k[:1])[0].detach().float().cpu() + 1e-6)   # k-space: log RSS
    show_m = lambda m: m[0, 0].expand(H, W).float().cpu()                        # mask: (1, W) → (H, W)
    ncols  = lambda m: int(m[0, 0, 0].sum())

    masks  = {"Bm": Bm, "A": A, "Ã = Bm·A": A_tilde}
    kspace = {"y": y, "Bm · y": Bm * y, f"x_t (t={t[0].item():.2f})": xt, "u": u, 
        f"tt * y (t={t[0].item():.2f})": tt * y, f"Bm * y + x0": Bm * y + x0}

    n = len(kspace)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
    # for a, (name, m) in zip(ax, masks.items()):
    #     a.imshow(show_m(m), cmap="gray", vmin=0, vmax=1)
    #     a.set_title(f"{name} ({ncols(m)}/{W} cols)"); a.axis("off")
    for a, (name, k) in zip(ax, kspace.items()):
        a.imshow(show_k(k), cmap="gray"); a.set_title(name); a.axis("off")
    plt.tight_layout(); plt.show()

    loss = (A * v - u).abs().pow(2).sum() / (A.sum() * C * H)
    baseline = u.abs().pow(2).sum() / (A.sum() * C * H)
    print(f"loss = {loss.item():.4f}   baseline = {baseline.item():.4f}   ratio = {(loss / baseline).item():.3f}")
    return loss
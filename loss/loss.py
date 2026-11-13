
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
    xt = tt * y + (1 - tt) * (Bm * y + A_tilde * x0)
    u = y - (Bm * y + A_tilde * x0)

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

    n_c = round(W * center_frac); c0, c1 = W // 2 - n_c // 2, W // 2 + n_c // 2
    Bm = (torch.rand(bs, 1, 1, W, device=device, generator=gen) > delta).float()
    Bm[..., c0:c1] = 1
    A_tilde = Bm * A
    t = torch.rand(bs, device=device, generator=gen) * (1 - 2 * eps) + eps
    x0 = torch.complex(torch.randn(y.shape, device=device, generator=gen),
                       torch.randn(y.shape, device=device, generator=gen)) / 2**0.5

    tt = t.view(-1, 1, 1, 1)
    xt = tt * y + (1 - tt) * (Bm * y + A_tilde * x0)
    u = y - (Bm * y + A_tilde * x0)

    inp = torch.cat([xt.real, xt.imag, A_tilde.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])

    show = lambda k: torch.log(rss(k[:1])[0].detach().float().cpu() + 1e-6)
    panels = {"y": y, "Bm · y": Bm * y, "Ã · x0": A_tilde * x0,
          f"x_t (t={t[0].item():.2f})": xt}
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for a, (name, k) in zip(ax, panels.items()):
        a.imshow(show(k), cmap="gray"); a.set_title(name); a.axis("off")
    plt.tight_layout(); plt.show()

    loss = (A * v - u).abs().pow(2).sum() / (A.sum() * C * H)
    assert (A * u - u).abs().max() == 0, \
        f"u has nonzero entries outside A: max |u·(1−A)| = {(u * (1 - A)).abs().max().item():.3e}"
    baseline = u.abs().pow(2).sum() / (A.sum() * C * H)
    print(f"loss = {loss.item():.4f}   baseline = {baseline.item():.4f}   ratio = {(loss / baseline).item():.3f}")
    return loss
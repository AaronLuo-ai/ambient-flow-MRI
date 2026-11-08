import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr       # add at the top of helper.py

@torch.no_grad()
def plot_recon(model, y, A, t0=1e-3):
    """One-step reconstruction of (y, A) with its PSNR. y: (1, C, H, W) k-space, A: (1, 1, 1, W)."""
    model.eval()
    device = next(model.parameters()).device                     # no global `device` needed
    y, A = y.to(device), A.to(device)
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)    # same scaling as training
    C, H, W = y.shape[1:]
    t = torch.full((1,), t0, device=device)

    inp = torch.cat([y.real, y.imag, A.expand(-1, 1, H, W)], dim=1)   # input: A x1 + mask A
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])
    x_hat = y + (1 - t0) * v                                     # one-step estimate, all lines
    model.train()
    zf, rec = rss(ifft2c(y)[0]), rss(ifft2c(x_hat)[0])
    print(f"shape y: {y.shape}, shape x_hat: {x_hat.shape}, shape zf: {zf.shape}, shape rec: {rec.shape}")
    show(torch.log(rss(y[0]) + 1e-9), torch.log(rss(x_hat[0]) + 1e-9), zf, rec)   # input k | recon k | input image | recon image

@torch.no_grad()
def compute_psnr(model, y, A, t0=1e-3):
    """Compute PSNR of one-step reconstruction without plotting."""
    model.eval()
    device = next(model.parameters()).device
    y, A = y.to(device), A.to(device)
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)
    C, H, W = y.shape[1:]
    t = torch.full((1,), t0, device=device)

    inp = torch.cat([y.real, y.imag, A.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])
    x_hat = y + (1 - t0) * v

    zf, rec = rss(ifft2c(y)[0]), rss(ifft2c(x_hat)[0])
    p = psnr(zf.cpu().numpy(), rec.cpu().numpy(), data_range=zf.max().item())
    return p
    
def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))

def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2,-1)), norm="ortho"), dim=(-2,-1))

def cartesian_mask(W, accel=4, center_frac=0.08, seed=0):
    rng = np.random.default_rng(seed)
    n_center = int(round(W * center_frac))
    m = rng.random(W) < (1 / accel - center_frac) / (1 - center_frac)   # random outer lines
    c = W // 2
    m[c - n_center // 2 : c + n_center // 2] = True                     # fully sampled center
    return torch.from_numpy(m).float()[None, None, :]                   # (1, 1, W)

def rss(z):
    """Combine coils: (C, H, W) complex → (H, W) magnitude."""
    return z.abs().pow(2).sum(0).sqrt()

def show(*imgs, size=3):
    """Plot 2D images only (no axes or colorbars)."""
    fig, ax = plt.subplots(1, len(imgs), figsize=(size * len(imgs), size))
    for a, im in zip(np.atleast_1d(ax), imgs):
        a.imshow(im.detach().float().cpu(), cmap="gray"); a.axis("off")
    plt.tight_layout(pad=0.2); plt.show()

def infinite(loader):
    while True:
        for batch in loader:
            yield batch
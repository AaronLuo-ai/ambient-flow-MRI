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
def compute_psnr_k_space(model, y, A, t0=1e-3, seed=0, plot=True):
    """Compute PSNR of one-step reconstruction in k-space from a noised input; optionally plot."""
    model.eval()
    device = next(model.parameters()).device
    y, A = y.to(device), A.to(device)
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)
    C, H, W = y.shape[1:]
    t = torch.full((1,), t0, device=device)

    g   = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(y.shape, dtype=y.dtype, device=device, generator=g)
    x_t = t0 * y + (1 - t0) * eps

    inp = torch.cat([x_t.real, x_t.imag, A.expand(-1, 1, H, W)], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp, t * 1000).sample
    v = torch.complex(out.float()[:, :C], out.float()[:, C:])
    x_hat = x_t + (1 - t0) * v

    gt  = y
    rec = x_hat
    gt_np, rec_np = gt.float().cpu().numpy(), rec.float().cpu().numpy()
    p   = psnr(gt_np, rec_np, data_range=gt_np.max())
    err = np.abs(gt_np - rec_np)

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(np.log(np.abs(gt_np[0,0]) + 1e-9), cmap="gray")
        ax[0].set_title("Ground truth k-space")
        ax[0].axis("off")
        ax[1].imshow(np.log(np.abs(rec_np[0,0]) + 1e-9), cmap="gray")
        ax[1].set_title(f"Reconstruction ({p:.2f} dB)")
        ax[1].axis("off")
        ax[2].imshow(np.log(err[0,0] + 1e-9), cmap="inferno")
        ax[2].set_title("Error")
        ax[2].axis("off")
        plt.tight_layout()
        plt.show()

    print(f"min error: {err.min():.4f}, max error: {err.max():.4f}, mean error: {err.mean():.4f}")
    return p


@torch.no_grad()
def plot_image(y, A, t=0.5, i=0, seed=0):
    """Plot mask, k-space and zero-filled recon for clean y and noised y_t = (1-t) y + t A eps."""
    t = min(max(t, 0.0), 1.0)
    y = y / ifft2c(y).abs().amax(dim=(1, 2, 3), keepdim=True)
    torch.manual_seed(seed)
    y_t = t * y + (1 - t) * A * torch.randn_like(y)

    kspace = lambda k: k.abs().pow(2).sum(0).sqrt().log1p().cpu()
    recon  = lambda k: rss(ifft2c(k[None]))[0].cpu()
    imgs = [A[i, 0].expand(y.shape[-2:]).float().cpu(),
            kspace(y[i]), kspace(y_t[i]), recon(y[i]), recon(y_t[i])]
    caps = ["Mask $A$", "$k$-space (clean)", f"$k$-space ($t$={t:.2f})",
            "Recon (clean)", f"Recon ($t$={t:.2f})"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, img, cap in zip(axes, imgs, caps):
        ax.imshow(img, cmap="gray"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(cap)
    plt.tight_layout(); plt.show()

    
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
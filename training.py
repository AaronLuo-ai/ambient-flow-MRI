# standard libraries
import os, sys, json, time, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
import copy
import inspect

# your project modules (run from the ambient_flow_MRI folder)
sys.path.append(os.getcwd())
from model.model import build_unet, UNetVelocity
from loss.loss import ambient_loss, ambient_loss_with_plot
from data.data import BrainAmbientFlowMatchingDataset
from util.helper import *

N_COILS = 16
DELTA, EPS, LR = 0.1, 1e-3, 1e-4
BATCH_SIZE = 4
N_EPOCHS, LOG_EVERY = 20, 20                    # validate once per epoch; log every LOG_EVERY batches
VAL_BATCHES = 25                                  # validate on a fixed subset for speed; None = all
ROOT = "/egr/research-slim/luozishe/MSM-MRI/ambient_flow_MRI"
save_path = f"{ROOT}/run";              os.makedirs(save_path, exist_ok=True)
ckpt_dir  = f"{ROOT}/ckpt_ambient_fm";  os.makedirs(ckpt_dir, exist_ok=True)
split_index = json.load(open(f"{ROOT}/split_16x320x320.json"))
 
LOG = os.path.join(save_path, "train_log.txt")

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def save_curves(hist):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["train_step"], hist["train_loss"], alpha=0.3, label="train (per step)")
    if len(hist["train_loss"]) >= 50:
        smooth = np.convolve(hist["train_loss"], np.ones(50) / 50, mode="valid")
        ax.plot(hist["train_step"][49:], smooth, label="train (50-step avg)")
    ax.plot(hist["val_step"], hist["val_loss"], "o-", label="val")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(save_path, "curves.png"), dpi=100); plt.close(fig)
 
@torch.no_grad()
def validate(model, val_loader, device, step):  # step = epoch here
    """Average val loss over a fixed subset, with fixed randomness so numbers are comparable across steps."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)       # same t, Bm, noise every validation
    losses = []
    for j, (vy, vA, _) in enumerate(val_loader):
        if VAL_BATCHES and j >= VAL_BATCHES:
            break
        vy, vA = vy.to(device), vA.to(device)
        if j == 0:                                            # one diagnostic figure per validation
            loss = ambient_loss_with_plot(model, vy, vA, delta=DELTA, eps=EPS, gen=gen)
            plt.savefig(os.path.join(save_path, f"val_epoch{step:03d}.png"), dpi=80)
            plt.close("all")
        else:
            loss = ambient_loss_with_plot(model, vy, vA, delta=DELTA, eps=EPS, gen=gen)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    train_set = BrainAmbientFlowMatchingDataset(index=split_index["train"], accel=4, cf=0.08)
    val_set   = BrainAmbientFlowMatchingDataset(index=split_index["val"],   accel=4, cf=0.08)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True,
                                               persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    log(f"train {len(train_set)} slices ({len(train_loader)} steps/epoch), val {len(val_set)} slices")
    model = build_unet(n_coils=N_COILS).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    hist = {"train_step": [], "train_loss": [], "val_step": [], "val_loss": []}
    best_val, best_epoch = float("inf"), -1
    step, t0 = 0, time.time()

    log(f"starting training for {N_EPOCHS} epochs")

    for epoch in range(1, N_EPOCHS + 1):
        # ---- train: one full pass over the data ----
        model.train()
        for y, A, _ in train_loader:
            step += 1
            y, A = y.to(device, non_blocking=True), A.to(device, non_blocking=True)
 
            train_loss = ambient_loss_with_plot(model, y, A, delta=DELTA, eps=EPS)
            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
 
            hist["train_step"].append(step); hist["train_loss"].append(train_loss.item())
            if step % LOG_EVERY == 0:
                log(f"epoch {epoch} step {step:6d}  train {np.mean(hist['train_loss'][-LOG_EVERY:]):.4f}  ({time.time() - t0:.0f}s)")
 
        # ---- validate: once per epoch, after the inner loop ----
        val_loss = validate(model, val_loader, device, epoch)
        hist["val_step"].append(step); hist["val_loss"].append(val_loss)
        epoch_train = np.mean(hist["train_loss"][-len(train_loader):])
        log(f"== epoch {epoch}/{N_EPOCHS}: train {epoch_train:.4f}  val {val_loss:.4f}")
 
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict(),
                    "val_loss": val_loss}, os.path.join(ckpt_dir, "last.pt"))
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                       os.path.join(ckpt_dir, "best.pt"))
            log(f"   saved best (val {best_val:.4f} at epoch {best_epoch})")
 
        json.dump(hist, open(os.path.join(save_path, "history.json"), "w"))
        save_curves(hist)
 
    log(f"done. best val {best_val:.4f} at epoch {best_epoch}")


def main():
    train()

if __name__ == "__main__":
    main()
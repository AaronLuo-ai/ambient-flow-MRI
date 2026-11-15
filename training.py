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
    t0 = time.time()

    log(f"starting training for {N_EPOCHS} epochs")

    train_losses, val_losses, train_steps, val_steps = [], [], [], []
    start_time = time.time()
    best_val, best_state, best_step = float("inf"), None, 0

    for epoch in range(1, N_EPOCHS + 1):
        # ---- train ----
        model.train()
        epoch_losses = []
        for y, A, _ in train_loader:
            y, A = y.to(device), A.to(device)
            train_loss = ambient_loss(model, y, A, delta=DELTA, eps=EPS)
            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_losses.append(train_loss.item())
        train_losses.append(np.mean(epoch_losses))               # mean over the epoch
        train_steps.append(epoch)
        print(f"epoch {epoch}: train {train_losses[-1]:.4f}")
        # ---- validate ----
        model.eval()
        with torch.random.fork_rng(devices=[device]), torch.no_grad():
            torch.manual_seed(0)                              # fixed seed inside this block only
            val_loss = np.mean([ambient_loss(model, vy.to(device), vA.to(device),
                                            delta=DELTA, eps=EPS).item()
                                for vy, vA, _ in val_loader])
        val_losses.append(val_loss)
        val_steps.append(epoch)
        print(f"epoch {epoch}: train {train_losses[-1]:.4f}  val {val_loss:.4f}")

        # ---- checkpoint: once per epoch, on the mean val loss ----
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(), "best_val": best_val, "best_epoch": best_epoch},
                    f"{ckpt_dir}/best.pt")
        
        hist = {"train_step": train_steps, "train_loss": train_losses,
        "val_step": val_steps, "val_loss": val_losses}
        json.dump(hist, open(f"{save_path}/history.json", "w"))
        save_curves(hist)
        t0 = time.time()

def main():
    train()

if __name__ == "__main__":
    main()
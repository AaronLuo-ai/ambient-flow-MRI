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

    

    train_losses, val_losses, train_steps, val_steps = [], [], [], []
    start_time = time.time()
    best_val, best_state, best_step = float("inf"), None, 0

    for epoch in range(1, N_EPOCHS + 1):
        # ---- train ----
        model.train()
        epoch_losses = []
        it = 0
        for Ay, A, _ in train_loader:
            it += 1
            Ay, A = Ay.to(device), A.to(device)
            train_loss = 0
            t = torch.rand(Ay.shape[0], device=device).view(-1, 1, 1, 1)
            if it % 20 == 0:
                train_loss = flow_loss(model, Ay, A, t, eps=EPS, plot=True, gen=None)[0]
            else:
                train_loss = flow_loss(model, Ay, A, t, eps=EPS, plot=False, gen=None)[0]
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
        torch.manual_seed(0)                                     # same randomness every epoch
        with torch.no_grad():
            val_loss = np.mean([flow_loss(model, vAy.to(device), vA.to(device),
                                        torch.rand(vAy.shape[0], device=device).view(-1, 1, 1, 1),
                                        eps=EPS, plot=False, gen=None)[0].item()
                                for vAy, vA, _ in val_loader])    # mean over all val batches
        val_losses.append(val_loss)
        val_steps.append(epoch)
        print(f"epoch {epoch}: train {train_losses[-1]:.4f}  val {val_loss:.4f}")

        # ---- checkpoint: once per epoch, on the mean val loss ----
        if val_loss < best_val:
            best_val, best_step = val_loss, epoch
            torch.save({"model": model.state_dict(), "best_val": best_val, "best_step": best_step},
                    f"{run_dir}/best.pt")

def main():
    train()

if __name__ == "__main__":
    main()
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
from loss.loss import ambient_flow_loss, flow_loss
from data.data import BrainAmbientFlowMatchingDataset
from util.helper import *

N_COILS = 16
DELTA, EPS, LR = 0.1, 1e-3, 1e-4
BATCH_SIZE = 4
N_EPOCHS, LOG_EVERY = 20, 20                    # validate once per epoch; log every LOG_EVERY batches
ROOT = "/egr/research-slim/luozishe/MSM-MRI/ambient_flow_MRI"
save_path = f"{ROOT}/run";              os.makedirs(save_path, exist_ok=True)
split_index = json.load(open(f"{ROOT}/split_16x320x320.json"))


def log(msg, path):
    line = f"{datetime.now():%H:%M:%S} | {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")

def save_curves(train_steps, train_losses, val_steps, val_losses, run_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_steps, train_losses, "o-", label="train")
    plt.plot(val_steps, val_losses, "o-", label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{run_dir}/curves.png", dpi=100); plt.close()
 
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(n_coils=N_COILS, with_mask=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    run_dir = os.path.join(save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    log_path = f"{run_dir}/train_log.txt"
    log(f"lr={LR} bs={BATCH_SIZE} epochs={N_EPOCHS}", log_path)

    train_set = BrainAmbientFlowMatchingDataset(index=split_index["train"], accel=4, cf=0.08)
    val_set   = BrainAmbientFlowMatchingDataset(index=split_index["val"],   accel=4, cf=0.08)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
     
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
                train_loss = flow_loss(model, Ay, A, t, eps=EPS, plot=False, gen=None)[0]
            else:
                train_loss = flow_loss(model, Ay, A, t, eps=EPS, plot=False, gen=None)[0]
            optim.zero_grad(set_to_none=True)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_losses.append(train_loss.item())
            if it % LOG_EVERY == 0:                                      # ← added
                log(f"epoch {epoch} it {it}: train {np.mean(epoch_losses[-LOG_EVERY:]):.4f}", log_path)
        train_losses.append(np.mean(epoch_losses))               # mean over the epoch
        train_steps.append(epoch)
        log(f"epoch {epoch}: train {train_losses[-1]:.4f}", log_path)
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
        log(f"epoch {epoch}: train {train_losses[-1]:.4f}  val {val_loss:.4f}  ({time.time() - start_time:.0f}s)", log_path)
        save_curves(train_steps, train_losses, val_steps, val_losses, run_dir)
        # ---- checkpoint: once per epoch, on the mean val loss ----
        if val_loss < best_val:
            best_val, best_step = val_loss, epoch
            torch.save({"model": model.state_dict(), "best_val": best_val, "best_step": best_step},
                    f"{run_dir}/best.pt")
            log(f"saved best: val {best_val:.4f} at epoch {best_step}", log_path)

def main():
    train()

if __name__ == "__main__":
    main()
# Ambient Flow Matching for MRI

Flow matching trained **only on undersampled k-space**, never on clean images. The model learns to fill in k-space lines that are hidden from it, using the fastMRI brain multicoil dataset.

## Method

Each training sample is a measured k-space $y = Ax_1$ with its sampling mask $A$. Training:

1. Drop extra lines with $B \sim \mathrm{Bernoulli}(1-\delta)$, giving $\tilde{A} = BA$ (the center is always kept).
2. Sample $t \sim \mathcal{U}[\varepsilon, 1-\varepsilon]$ and noise $x_0 \sim \mathcal{CN}(0, I)$ in k-space.
3. Straight path: $x_t = t\,Ax_1 + (1-t)\,\tilde{A}(x_1 + x_0)$.
4. Target velocity: $u = Ax_1 - \tilde{A}(x_1 + x_0)$.
5. Loss on $A$'s lines only: $\mathcal{L} = \lVert A\,u_\theta(x_t, \tilde{A}, t) - u \rVert^2$.

The network sees $[\mathrm{Re}(x_t), \mathrm{Im}(x_t), \tilde{A}]$ (33 channels) and predicts the k-space velocity (32 channels).

## Repository structure

```
ambient_flow_MRI/
├── data/data.py        # BrainAmbientFlowMatchingDataset: returns (A x1, A, index)
├── loss/loss.py        # ambient_loss: the training objective above
├── model/model.py      # build_unet (diffusers UNet2DModel), UNetVelocity (ODESolver wrapper)
├── util/helper.py      # fft2c / ifft2c, cartesian_mask, rss, show, plot_recon, infinite
├── train.ipynb         # training loop, validation, visualization, checkpoints
├── requirements.txt    # Python dependencies
├── split_16x320x320.json        # train / val / test split by subject
├── index_16x320x320.json        # slices with k-space shape (16, 320, 320)
├── kspace_shape_table.csv       # k-space shapes across the dataset
└── ckpt_ambient_fm/             # saved checkpoints (step_*.pt)
```

Each folder needs an empty `__init__.py` so it can be imported.

## Data

- fastMRI brain multicoil training set, one `.npz` file per slice: `kspace` (coils, H, W), `target` (RSS ground truth), metadata.
- Only subjects with k-space shape **(16, 320, 320)** are used (726 subjects, 11,438 slices), so batches can be stacked.
- The split is **by subject**, 80 / 10 / 10 (580 / 73 / 73 subjects; 9,134 / 1,150 / 1,154 slices), with a fixed seed.
- Acceleration mask $A$: Cartesian, `accel=4`, `center_frac=0.08` (26 fully sampled center lines), fixed per slice by its index.

## Setup

```bash
conda create -n flow_matching python=3.9
conda activate flow_matching
pip install -r requirements.txt
```

## Training

Open `train.ipynb` and run the cells top to bottom. Main settings:

| Parameter | Value | Meaning |
|---|---|---|
| `N_COILS` | 16 | coils per slice |
| `DELTA` | 0.1 | fraction of $A$'s lines dropped by $B$ |
| `EPS` | 1e-3 | $t \in [\varepsilon, 1-\varepsilon]$ |
| `LR` | 1e-4 | AdamW learning rate |
| `BATCH_SIZE` | 4 | slices per batch |
| `MAX_STEPS` | 10000 | training steps |

Every `print_every` steps, the notebook shows the train and validation loss curves and the reconstruction of a fixed validation slice with its PSNR. Checkpoints are saved to `ckpt_ambient_fm/` every `ckpt_every` steps.

## Notes

- Use `%load_ext autoreload` / `%autoreload 2` in the notebook so edits to the `.py` modules load without restarting the kernel.
- The current objective only supervises lines inside $A$. Recovering lines outside $A$ at test time relies on the model generalizing beyond its training task. Evaluate this on the test set against `target`.
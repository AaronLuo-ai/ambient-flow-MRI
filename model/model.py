import torch
from diffusers import UNet2DModel
from flow_matching.utils import ModelWrapper

def build_unet(n_coils=16):
    return UNet2DModel(
        sample_size=[320, 320],
        in_channels=2 * n_coils + 1,
        out_channels=2 * n_coils,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

class UNetVelocity(ModelWrapper):
    def __init__(self, unet, mask):
        super().__init__(unet)
        self.mask = mask

    def forward(self, x, t, **extras):
        bs, _, H, W = x.shape
        t = t.reshape(-1).expand(bs)
        m = self.mask.expand(bs, 1, H, W)
        return self.model(torch.cat([x, m], 1), t * 1000).sample
import torch
import pytorch_lightning as pl
from torch.functional import align_tensors

import typing as T


class PairedConvolution(pl.LightningModule):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super(self.__class__, self).__init__()
        self._conv = torch.nn.Conv2d(in_channels, out_channels*2, kernel_size, **kwargs)
        self._offsets = torch.nn.Parameter((torch.rand(out_channels, 2) - 0.5) * 2).to(self.device)
        self._grids: T.Dict[torch.Size, torch.Tensor] = dict()

    def forward(self, x):
        """
        TODO(pradeepr): unittest grid generation
        TODO(pradeepr): unittest pair aggregation
        """
        phi = self._conv(x)  # conv filter responses (feature maps)
        B, C, H, W = phi.size()
        if (B, H, W) not in self._grids:
            rgrid, cgrid = torch.meshgrid(torch.arange(H), torch.arange(W))
            self._grids[(B, H, W)] = torch.stack((rgrid.repeat(B, 1, 1), cgrid.repeat(B, 1, 1)), -1).to(self.device)
            print('Generated grid of size: ', self._grids[(B, H, W)].shape)
        grid = self._grids[(B, H, W)]

        # Output will have `C//2` channels as the convolutional output
        # `phi` with `C` channels. This is because adjacent conv
        # responses will be aggregated, pair-wise, into one response.
        y = torch.zeros(B, C//2, H, W)

        # Iterate pair-by-pair (group), sample the second image
        # according to its corresponding offset and combine with the
        # first image using elementwise product. This operation is
        # intended to capture the "relative-phase" between adjacent
        # features
        for g in range(C//2):
            p, q = g*2, g*2+1  # g-th pair
            offset_grid = grid + self._offsets[g]
            offset_sampled_q = torch.nn.functional.grid_sample(phi[:, q:q+1, :, :], offset_grid, align_corners=False)
            y[:, g, :, :] = (phi[:, p:p+1, :, :] * offset_sampled_q).squeeze()

        return y

x = torch.rand(5, 1, 28, 28)
pc = PairedConvolution(1, 3, 5, padding=2)
pc.forward(x)
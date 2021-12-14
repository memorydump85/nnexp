import datetime
import enum
from pathlib import Path
import subprocess
import typing as T

import numpy as np
from numpy.core.numerictypes import ScalarType
import pytorch_lightning as pl

from torch.nn import functional as F
import torch
from torchvision.utils import make_grid, save_image

from scanwise_data import ScanwiseDetectorDataModule, ScanwiseDetectorDataset
from scanwise_config import ModelConfig


def tensor_to_img(tensor, ch=0, allkernels=True, nrow=1, padding=1):
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
    return make_grid(tensor, nrow=nrow, normalize=True, padding=padding)


class CenteredConvolutionLayer(pl.LightningModule):
    """Center per-channel values, by subtracting out mean, for each
    position of the kernel and then compute ReLU activation on a linear
    combination of centered and un-centered channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        # TODO: wraparound scan for objects at border?
        self._conv = torch.nn.Conv1d(in_channels=in_channels*2, out_channels=out_channels,
                                     kernel_size=kernel_size, padding=padding, dilation=dilation)
        self._avg_kernel = torch.ones(1, 1, kernel_size) / kernel_size
        self._activation = torch.nn.ReLU()
        self.add_module('_conv', self._conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W = x.shape
        # De-bias channels and concat to input-channels to make 2*C channels
        avg_filtered = torch.conv1d(x.view(B * C, 1, W), self._avg_kernel, bias=None,
                                    stride=self._conv.stride, padding=self._conv.padding,
                                    dilation=self._conv.dilation).view(B, C, W)
        x = torch.cat([x, x - avg_filtered], dim=1)
        return self._activation(self._conv(avg_filtered))


GTChannel = ScanwiseDetectorDataset.GroundTruthChannel

def get_channel_BHCW(v: torch.Tensor, ch: int) -> torch.Tensor:
    return v[:, :, ch:ch+1, :]

class ScanwiseDetectorModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        NUM_FILTERS=4

        self._scaled_convolutions : T.Dict[str, CenteredConvolutionLayer] = {}
        for w in (2, 4, 8, 16, 32, 64):
            if ModelConfig.use_centered_convolutions:
                self._scaled_convolutions[f'conv{w}'] = CenteredConvolutionLayer(
                    in_channels=3, out_channels=NUM_FILTERS, kernel_size=9, padding=w*4, dilation=w)
            else:
                # TODO: wraparound scan for objects at border?
                self._scaled_convolutions[f'conv{w}'] = torch.nn.Sequential(torch.nn.Conv1d(
                    in_channels=3, out_channels=NUM_FILTERS, kernel_size=9, padding=w*4, dilation=w),
                    torch.nn.LeakyReLU())

        for name, layer in self._scaled_convolutions.items():
            self.add_module(name, layer)
        self._linear_bottleneck = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=NUM_FILTERS*6, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cat([conv(x) for conv in self._scaled_convolutions.values()], dim=1)
        return self._linear_bottleneck(u)

    class _LossComputationResult(T.NamedTuple):
        loss: float
        y_hat: torch.Tensor
        t: torch.Tensor

    def loss(self, batch, batch_idx) -> _LossComputationResult:
        x, y = batch
        B, H, C, W = x.shape
        y_hat = self(x.view(B*H, C, W)).view(B, H, 1, W)
        t = get_channel_BHCW(y, GTChannel.BOX_INDEX)
        loss = F.binary_cross_entropy_with_logits(y_hat, t)
        return self._LossComputationResult(loss, y_hat, t)

    def training_step(self, batch, batch_idx):
        r = self.loss(batch, batch_idx)
        self.log('train_loss', r.loss)
        return {'loss': r.loss}

    def training_epoch_end(self, _outputs) -> None:
        self.log('learning_rate', self._scheduler.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        r = self.loss(batch, batch_idx)
        return {'val_loss': r.loss,
                'y_hat': torch.sigmoid(r.y_hat).transpose(1, 2).contiguous(),  # BHCW -> BCHW
                'y': r.t.transpose(1, 2).contiguous()}

    def validation_epoch_end(self, outputs):
        y_hat = tensor_to_img(outputs[0]['y_hat'])
        y = tensor_to_img(outputs[0]['y'])
        save_image(y_hat, '/tmp/y_hat.png')
        save_image(y, '/tmp/y.png')
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [500, 3000], verbose=True)
        # self._scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=0.01, total_steps=100)
        return [self._optimizer], [self._scheduler]


def get_git_info():

    def check_output(cmd):
        return subprocess.check_output(cmd).decode('utf-8').strip()

    cmd = ['git', '-C', str(Path(__file__).parent)]
    sha = check_output(cmd + ['rev-parse', 'HEAD'])
    commit_info = check_output(cmd + ['log', '-1', '--oneline'])
    diff = check_output(cmd + ['diff', '--stat']) + "\n\n" + check_output(cmd + ['--no-pager', 'diff'])

    return sha, commit_info, diff


def get_next_minor_version(dir_: Path, prefix: str) -> str:
    existing = list(dir_.glob(f'**/{prefix}.*'))
    print(dir_, prefix, existing)
    if not existing:
        max_version = -1
    else:
        max_version = max(int(x.name.split('.')[1]) for x in existing)
    return f'{max_version + 1:05d}'


def main():
    if ModelConfig.use_manual_seed:
        np.random.seed(313424)
        torch.manual_seed(313424)

    sha, commit_info, diff = get_git_info()
    model = ScanwiseDetectorModel()
    data_module = ScanwiseDetectorDataModule(Path('/tmp/waymo_od_lidar'))

    log_dir = Path(__file__).parent / f'_lightning_logs'
    model_name = model.__class__.__name__
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir,
                                          name=model_name,
                                          version=sha[:7] + '.' + get_next_minor_version(log_dir, sha[:7]))
    logger.experiment.add_text('diff', f'# {commit_info}\n\n<pre>\n{diff}\n</pre>\n')
    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None,
                         accelerator='ddp' if torch.cuda.is_available() else None,
                         logger=logger)  #, profiler='advanced')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

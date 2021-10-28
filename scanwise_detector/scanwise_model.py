from pathlib import Path
import typing as T
import platform

import numpy as np
import pytorch_lightning as pl

from torch.nn import functional as F
import torch
from torchvision.utils import make_grid, save_image


def tensor_to_img(tensor, ch=0, allkernels=True, nrow=1, padding=1):
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
    return make_grid(tensor, nrow=nrow, normalize=True, padding=padding)


class ScanwiseDetectorDataset(torch.utils.data.Dataset):
    def __init__(self, path_prefixes: T.Iterable[Path]) -> None:
        super().__init__()
        self._path_prefixes = list(path_prefixes)

    def __getitem__(self, i: int) -> torch.Tensor:
        relevant_dims = (0, 1, 2, 4)  # range, intensity, elongation, (no-label-zone), box-index
        d = torch.from_numpy(np.load(self._path_prefixes[i].with_suffix('.npy'))[relevant_dims, ...])
        d[0, ...] = (d[0, ...].clamp(0, np.inf) - 37.5) / 37.5     # normalize to [0, 1]
        d[1, ...] = (d[1, ...].clamp(0, np.inf) - 25000) / 25000   # normalize to [0, 1]
        d[2, ...] = (d[2, ...].clamp(0, np.inf) - 0.75) / 0.75     # normalize to [0, 1]
        d[3, ...] = d[3, ...].clamp(-1, 0) + 1    # Convert box index to box yes/no
        # print(d[0, ...].max(), d[1, ...].max(), d[2, ...].max(), d[3, ...].max())
        # print(d[0, ...].min(), d[1, ...].min(), d[2, ...].min(), d[3, ...].min())
        # print()
        return d

    def __len__(self):
        return len(self._path_prefixes)

mini = (platform.node() == 'p50')

class DataModuleConfig:
    batch_size: int =  8 if mini else 32


class ScanwiseDetectorDataModule(pl.LightningDataModule):

    class Config:
        batch_size: int = 8

    def __init__(self, data_root_dir: Path, batch_size: int=8):
        super().__init__()
        self._train = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])
        self._val = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])
        self._test = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])

    def prepare_data(self):
        pass

    def setup(self, stage: T.Optional[str] = None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train, batch_size=DataModuleConfig.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val, batch_size=DataModuleConfig.batch_size, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test, batch_size=DataModuleConfig.batch_size, pin_memory=True)


class ScanwiseDetectorModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        """
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2),
            torch.nn.ConstantPad2d((0, 2), 0))
        """
        """
        NUM_FILTERS=20
        self._scaled_convolutions = { f'conv{w}':
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=3, out_channels=NUM_FILTERS, kernel_size=w + 1, padding=w // 2),
                torch.nn.LeakyReLU())
                    for w in (4, 8, 16, 32, 64) }  # TODO: wraparound scan for objects at border
        for name, layer in self._scaled_convolutions.items():
            self.add_module(name, layer)

        self._linear_bottleneck = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=NUM_FILTERS*5, out_channels=1, kernel_size=1, padding=0),
        )
        """
        NUM_FILTERS=4
        self._scaled_convolutions = { f'conv{w}':
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=3, out_channels=NUM_FILTERS, kernel_size=9, padding=w*4, dilation=w),
                torch.nn.LeakyReLU())
                    for w in (2, 4, 8, 16, 32, 64) }  # TODO: wraparound scan for objects at border
        for name, layer in self._scaled_convolutions.items():
            self.add_module(name, layer)
        self._linear_bottleneck = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=NUM_FILTERS*6, out_channels=1, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.cat([conv(x) for conv in self._scaled_convolutions.values()], dim=1)
        return self._linear_bottleneck(u)

    @staticmethod
    def batch_images_to_scans(batch: torch.Tensor) -> torch.Tensor:
        """Separate out the rows of each image (2D) in a `batch` to its
        own separate signal (1D).
        """
        B, C, H, W = batch.shape
        return batch.transpose(1, 2).reshape(B*H, C, W)  # BCHW -> (B*H)*C*W

    @staticmethod
    def batch_scans_to_images(batch: torch.Tensor, im_height: int) -> torch.Tensor:
        """Combine separate scans (1D) into an image (2D). This is the inverse
        of `batch_images_to_scans`.
        """
        BH, C, W = batch.shape
        return batch.reshape(-1, im_height, C, W).transpose(2, 1)  # (B*H)*C*W -> BCHW

    def training_step(self, batch, batch_idx):
        _B, _C, H, _W = batch.shape
        batch = __class__.batch_images_to_scans(batch)
        print(batch.is_contiguous())
        x, y = batch[:, :3, :], batch[:, 3:, :]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, _outputs) -> None:
        self.log('learning_rate', self._scheduler.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        _B, _C, H, _W = batch.shape
        batch = __class__.batch_images_to_scans(batch)
        x, y = batch[:, :3, :], batch[:, 3:, :]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = __class__.batch_scans_to_images(y_hat, H)
        y = __class__.batch_scans_to_images(y, H)
        return {'val_loss': loss, 'y_hat': torch.sigmoid(y_hat), 'y': y}

    def validation_epoch_end(self, outputs):
        y_hat = tensor_to_img(outputs[0]['y_hat'])
        y = tensor_to_img(outputs[0]['y'])
        save_image(y_hat, '/tmp/y_hat.png')
        save_image(y, '/tmp/y.png')
        # self.logger.experiment.add_image(f'predictions', im, self.current_epoch)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [500, 3000], verbose=True)
        # self._scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=0.01, total_steps=100)
        return [self._optimizer], [self._scheduler]


def main():
    model = ScanwiseDetectorModel()
    data_module = ScanwiseDetectorDataModule(Path('/tmp/waymo_od_lidar'))

    model.training_step(next(iter(data_module.train_dataloader())), 0)

    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None,
                         accelerator='ddp' if torch.cuda.is_available() else None,
                         profiler='advanced')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

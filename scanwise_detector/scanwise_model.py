from pathlib import Path
import typing as T

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


class ScanwiseDetectorDataModule(pl.LightningDataModule):
    def __init__(self, data_root_dir: Path, batch_size: int=8):
        super().__init__()
        self._batch_size = batch_size
        self._train = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])
        self._val = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])
        self._test = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])

    def prepare_data(self):
        pass

    def setup(self, stage: T.Optional[str] = None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._train, batch_size=self._batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._val, batch_size=self._batch_size, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self._test, batch_size=self._batch_size, pin_memory=True)


class ScanwiseDetectorModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :3, :], batch[:, 3:, :]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, _outputs) -> None:
        self.log('learning_rate', self._scheduler.get_last_lr()[0], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :3, :], batch[:, 3:, :]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return {'val_loss': loss, 'y_hat': F.sigmoid(y_hat), 'y': y}

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
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [300, 3000], verbose=True)
        # self._scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=0.01, total_steps=100)
        return [self._optimizer], [self._scheduler]


def main():
    model = ScanwiseDetectorModel()
    data_module = ScanwiseDetectorDataModule(Path('/tmp/waymo_od_lidar'))

    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None,
                         accelerator='ddp' if torch.cuda.is_available() else None)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

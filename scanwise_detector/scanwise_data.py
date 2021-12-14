import enum
import functools
from pathlib import Path
import typing as T

import numpy as np
import torch
import pytorch_lightning as pl

from scanwise_config import DataModuleConfig


class ScanwiseDetectorDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` that loads data from a list of
    `".npy"` file paths.

    Args:
        path_prefixes: .npy file paths containing exported range image
        and ground truth data.
    """
    class _ExportedChannel(enum.IntEnum):
        """Numerical indices for the channels of the dataset as saved
        from the exported numpy datafile.
        """
        RANGE = 0
        INTENSITY = 1
        ELONGATION = 2
        _NUM_INPUT_CHANNELS = 3  # Following channels contain ground-truth.
        NO_LABEL_ZONE = 3
        BOX_INDEX = 4
        BOX_CENTER_X = 5
        BOX_CENTER_Y = 6
        BOX_CENTER_Z = 7
        BOX_LENGTH = 8
        BOX_WIDTH = 9
        BOX_HEIGHT = 10
        BOX_HEADING = 11

    class SignalChannel(enum.IntEnum):
        """Numerical indices for the input signal channels of the dataset."""
        RANGE = 0
        INTENSITY = 1
        ELONGATION = 2

    class GroundTruthChannel(enum.IntEnum):
        """Numerical indices for the ground-truth channels of the dataset."""
        NO_LABEL_ZONE = 0
        BOX_INDEX = 1
        BOX_CENTER_X = 2
        BOX_CENTER_Y = 3
        BOX_CENTER_Z = 4
        BOX_LENGTH = 5
        BOX_WIDTH = 6
        BOX_HEIGHT = 7
        BOX_HEADING = 8

    def __init__(self, path_prefixes: T.Iterable[Path]) -> None:
        super().__init__()
        self._input_paths = list(path_prefixes)

    def __getitem__(self, i: int) -> torch.Tensor:
        CH = self._ExportedChannel
        d = np.load(self._input_paths[i])
        d[CH.RANGE, ...] = (d[CH.RANGE, ...].clip(0, np.inf) - 37.5) / 37.5  # normalize to [0, 1]
        d[CH.INTENSITY, ...] = (d[CH.INTENSITY, ...].clip(0, np.inf) - 25000) / 25000  # normalize to [0, 1]
        d[CH.ELONGATION, ...] = (d[CH.ELONGATION, ...].clip(0, np.inf) - 0.75) / 0.75  # normalize to [0, 1]
        d[CH.BOX_INDEX, ...] = d[CH.BOX_INDEX, ...].clip(-1, 0) + 1  # Convert box index to box yes/no
        return (torch.from_numpy(d[:CH._NUM_INPUT_CHANNELS, ...]).contiguous(),
                torch.from_numpy(d[CH._NUM_INPUT_CHANNELS:, ...]).contiguous())

    def __len__(self):
        return len(self._input_paths)


class ScanwiseDetectorDataModule(pl.LightningDataModule):
    """A `pytorch_lightning.LightningDataModule` that encapsulates train,
    validation and test `ScanwiseDetectorDataset`s.
    """
    def __init__(self, data_root_dir: Path, batch_size: int=DataModuleConfig.batch_size):
        super().__init__()
        if DataModuleConfig.dev_trial:
            ds = ScanwiseDetectorDataset(list(data_root_dir.glob('*/*.npy'))[:8])
            self._train = self._val = self._test = ds
        else:
            paths = list(data_root_dir.glob('*/*.npy'))
            np.random.shuffle(paths)
            validation_start = int(len(paths) * 0.7)
            test_start = int(len(paths) * 0.9) if DataModuleConfig.use_test_split else len(paths)
            assert test_start > validation_start > 0
            self._train = ScanwiseDetectorDataset(paths[:validation_start])
            self._val = ScanwiseDetectorDataset(paths[validation_start:test_start])
            self._test = ScanwiseDetectorDataset(paths[test_start:])
        print('\nDataModule:')
        print('-----------')
        print(f'    Train: {len(self._train)} instances')
        print(f'      Val: {len(self._val)} instances')
        print(f'     Test: {len(self._test)} instances')
        print('')

    def prepare_data(self):
        pass

    def setup(self, stage: T.Optional[str] = None):
        pass
    
    @staticmethod
    def collate_batch(batch: T.List[torch.Tensor]) -> T.Tuple[torch.Tensor, torch.Tensor]:
        x, y = torch.stack([x for x, _ in batch]), torch.stack([y for _, y in batch])
        return x.transpose(1, 2).contiguous(), y.transpose(1, 2).contiguous()  # BCHW -> BHCW

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train, batch_size=DataModuleConfig.batch_size, shuffle=True, pin_memory=True,
            collate_fn=self.__class__.collate_batch)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val, batch_size=DataModuleConfig.batch_size, pin_memory=True,
            collate_fn=self.__class__.collate_batch)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test, batch_size=DataModuleConfig.batch_size, pin_memory=True,
            collate_fn=self.__class__.collate_batch)

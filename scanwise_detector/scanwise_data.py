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
        check_assumptions: if > 0, explicitly checked for assumptions
            in the loaded data. Higher values check more assumptions
            (or) are more verbose.
    """
    class _ExportedChannel(enum.IntEnum):
        """Numerical indices for the channels of the dataset as saved
        from the exported numpy datafile.
        """
        RANGE = 0
        INTENSITY = 1
        ELONGATION = 2
        _NUM_INPUT_CHANNELS = 3  # Following channels contain ground-truth.
        X = 3
        Y = 4
        Z = 5
        NO_LABEL_ZONE = 6
        BOX_INDEX = 7
        BOX_CENTER_X = 8
        BOX_CENTER_Y = 9
        BOX_CENTER_Z = 10
        BOX_LENGTH = 11
        BOX_WIDTH = 12
        BOX_HEIGHT = 13
        BOX_HEADING = 14
        _NUM_CHANNELS = 15

    class SignalChannel(enum.IntEnum):
        """Numerical indices for the input signal channels of the dataset."""
        RANGE = 0
        INTENSITY = 1
        ELONGATION = 2

    class GroundTruthChannel(enum.IntEnum):
        """Numerical indices for the ground-truth channels of the dataset."""
        X = 0
        Y = 1
        Z = 2
        NO_LABEL_ZONE = 3
        IS_IN_BOX = 4
        BOX_CENTER_OFFSET_X = 5
        BOX_CENTER_OFFSET_Y = 6
        BOX_CENTER_OFFSET_Z = 7
        BOX_LENGTH = 8
        BOX_WIDTH = 9
        BOX_HEIGHT = 10
        BOX_HEADING = 11

    def __init__(self, path_prefixes: T.Iterable[Path], check_assumptions: int=0) -> None:
        super().__init__()
        self._input_paths = list(path_prefixes)
        self._check_assumptions = check_assumptions

    def __getitem__(self, i: int) -> torch.Tensor:
        CH = self._ExportedChannel
        d = np.load(self._input_paths[i])
        d[CH.RANGE, ...] = (d[CH.RANGE, ...].clip(0, np.inf) - 37.5) / 37.5  # normalize to [0, 1]
        d[CH.INTENSITY, ...] = (d[CH.INTENSITY, ...].clip(0, np.inf) - 25000) / 25000  # normalize to [0, 1]
        d[CH.ELONGATION, ...] = (d[CH.ELONGATION, ...].clip(0, np.inf) - 0.75) / 0.75  # normalize to [0, 1]
        d[CH.BOX_INDEX, ...] = d[CH.BOX_INDEX, ...].clip(-1, 0) + 1  # box-index -> is-in-box
        not_an_object_mask = d[CH.BOX_INDEX, ...] == 0
        if self._check_assumptions:
            for ch in CH.BOX_LENGTH, CH.BOX_WIDTH, CH.BOX_HEIGHT:
                assert (not_an_object_mask == (d[ch, ...] < 0)).all(), \
                    f'Not an object mask check failed for {ch.name}'
        # Convert center coordinates to offsets (box-center-* -> box-center-offset-*)
        for box_info_ch, ch in (CH.BOX_CENTER_X, CH.X), (CH.BOX_CENTER_Y, CH.Y), (CH.BOX_CENTER_Z, CH.Z):
            d[box_info_ch, ...] -= d[ch, ...]
            d[box_info_ch, not_an_object_mask] = -np.inf
        if self._check_assumptions > 1:
            print('Range:\t', d[CH.RANGE, ...].min(), d[CH.RANGE, ...].max())
            print('Inten:\t', d[CH.INTENSITY, ...].min(), d[CH.INTENSITY, ...].max())
            print('Elong:\t', d[CH.ELONGATION, ...].min(), d[CH.ELONGATION, ...].max())
            valid = d[CH.RANGE, ...] >= 0
            print('    x:\t', d[CH.X, valid].min(), d[CH.X, valid].max())
            print('    y:\t', d[CH.Y, valid].min(), d[CH.Y, valid].max())
            print('    z:\t', d[CH.Z, valid].min(), d[CH.Z, valid].max())
            print('box.x:\t', d[CH.BOX_CENTER_X, ...].min(), d[CH.BOX_CENTER_X, ...].max())
            print('box.y:\t', d[CH.BOX_CENTER_Y, ...].min(), d[CH.BOX_CENTER_Y, ...].max())
            print('box.z:\t', d[CH.BOX_CENTER_Z, ...].min(), d[CH.BOX_CENTER_Z, ...].max())
            print('box.h:\t', d[CH.BOX_HEADING, ...].min(), d[CH.BOX_HEADING, ...].max())
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
        # Our model works on individual LiDAR scans. Hence the `(H)eight`
        # dimension of the input tensor should be merged with the
        # `(B)atch` dimension during prediction. We would still want to
        # hang on to the original dimensions for visulizing the result
        # as an image, if needed.
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

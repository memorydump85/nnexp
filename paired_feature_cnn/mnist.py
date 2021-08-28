import os

import numpy as np
from scipy import signal
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from paired_convolution import PairedConvolution


def tensor_to_img(tensor, ch=0, allkernels=True, nrow=16, padding=1):
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    return make_grid(tensor, nrow=nrow, normalize=True, padding=padding)


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        self.filters1 = PairedConvolution(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Sequential(
            self.filters1,
            nn.SELU(),
            nn.MaxPool2d(kernel_size=7),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.out(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, _outputs) -> None:
        self.logger.experiment.add_image(f'filters', tensor_to_img(self.filters1._conv.weight), self.current_epoch)
        self.log('learning_rate', self._scheduler.get_last_lr()[0], prog_bar=True)
        print(self.filters1._offsets)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def prepare_data(self):
        transform=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=1024, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=1024, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1024, pin_memory=True)

    def configure_optimizers(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, [30, 300], verbose=True)
        # self._scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=0.01, total_steps=100)
        # return [self._optimizer], [self._scheduler]
        return self._optimizer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", help="Restore training from checkpoint", default=None)
    args = parser.parse_args()

    model = LightningMNISTClassifier()
    trainer = pl.Trainer(resume_from_checkpoint=args.ckpt,
                            gpus=-1 if torch.cuda.is_available() else None,
                            accelerator='ddp')
    trainer.fit(model)

if __name__ == '__main__':
    main()

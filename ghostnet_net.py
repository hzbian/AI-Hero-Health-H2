import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import h5py
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from argparse import ArgumentParser
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append('..')

#sns.set(style="darkgrid")

class H5Dataset(Dataset):
    def __init__(self, path, load_num=100, transform=None, target_transform=None):
        self.transform = transform
        self.load_num = load_num
        self.target_transform = target_transform
        self.path = path
        self.open_hdf5()
        self.x = torch.tensor(self.img_hdf5['imgs'][:load_num])
        self.y = torch.tensor(self.img_hdf5['labels'][:load_num])
        
    def __len__(self):
        return self.load_num

    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.path, 'r')

    def __getitem__(self, idx):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        y = self.y[idx]
        x = self.x[idx]

        if self.transform:
            y = self.transform(y)
        return x.float(), y.float()

class GhostNetNet(pl.LightningModule):
    def __init__(self, name:str='test', layer_size:int=5, blow:float=0., shrink_factor:str='log', learning_rate:float=0.001, gpus:int=0, optimizer:str='adam'):
        super(GhostNetNet, self).__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()

        self.net = self.create_sequential(1000, 1, self.hparams.layer_size, blow=self.hparams.blow, shrink_factor=self.hparams.shrink_factor)
        print(self.net)

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def create_sequential(self, input_length, output_length, layer_size, blow=0, shrink_factor="log"):
        layers = [input_length]
        blow_disabled = blow == 1 or blow == 0
        if not blow_disabled:
            layers.append(input_length*blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(math.log(layers[-1], 10), math.log(output_length,10), steps=layer_size+2-len(layers), base=10).long()
            # make sure the last element is correct, even though rounding
            add_layers[-1] = output_length
        elif shrink_factor == "lin":
            add_layers = torch.linspace(layers[-1], output_length, steps=layer_size+2-len(layers)).long()
        else:
            shrink_factor = float(shrink_factor)
            new_length = layer_size+1-len(layers)
            add_layers = (torch.ones(new_length)*layers[-1] * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))).long()
            layers = torch.cat((torch.tensor([input_length]), add_layers))
            layers = torch.cat((layers, torch.tensor([output_length])))
    
        if not blow_disabled:
            layers = torch.tensor([layers[0]])
            layers = torch.cat((layers, add_layers))
        else:
           layers = add_layers
           layers[0] = input_length

        nn_layers = []
        for i in range(len(layers)-1):
            nn_layers.append(nn.Linear(layers[i].item(), layers[i+1].item()))
            if not i == len(layers)-2:
                nn_layers.append(nn.ReLU())
                #nn_layers.append(nn.BatchNorm1d(layers[i+1].item()))
        return nn.Sequential(*nn_layers)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y, y_hat)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y, y_hat)
        return {'s_test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['s_test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        return  avg_loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y, y_hat)
        y_hat = (y_hat > 0.5).float()
        accuracy = (y_hat == y).float().mean() * 100.
        return {'s_val_loss': loss, 'accuracy': accuracy}


    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['s_val_loss'] for x in outputs]).mean()
        val_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        self.log('val_loss', val_loss)
        self.log('val_accuracy', val_accuracy, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]
        elif self.hparams.optimizer == 'sgd':
            return [optim.SGD(model.parameters(), lr=self.hparams.learning_rate, momentum=0.9)]

class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 2,
        num_workers: int = 0,
        data_root: str = 'test.h5',
        sample_size = 100,
    ):
        super().__init__()
        #self.save_hyperparameters()
        self.target_transform =  transforms.Compose([
            transforms.Lambda(lambda x : x.flatten())
            ])
        self.dataset = H5Dataset(data_root, target_transform=self.target_transform, load_num=sample_size)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = False

        train_size = int(0.6 * len(self.dataset))
        val_size = int(0.4 * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])
 
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.on_gpu)

if __name__ == '__main__':
    try:
        name = sys.argv[sys.argv.index("--model.name") + 1]
    except ValueError:
        name = [i for i in sys.argv if '--model.name=' in i][0][13:]
    logger = TensorBoardLogger(save_dir='lightning_logs', name=name)
    cli = LightningCLI(GhostNetNet, MyDataModule, seed_everything_default=42, trainer_defaults={"logger": logger}, save_config_overwrite=True)

import logging

import h5py
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class H5Dataset(Dataset):
    def __init__(self, path, load_num=100, transform=None,
                 target_transform=None,
                 mean=0., std=1.):
        self.transform = transform
        self.load_num = load_num
        self.target_transform = target_transform
        self.path = path
        self.open_hdf5()
        self.x = torch.tensor(self.img_hdf5['imgs'][:load_num])
        if mean is None or std is None:
            mean = self.x.mean(dim=0)
            std = self.x.std(dim=0)
            std[std == 0.] = 1.

        self.x = (self.x - mean) / std

        mn = self.x.mean(dim=0)
        logger.info(f'{mn.mean()=}')
        logger.info(f'{mn.std()=}')
        sd = self.x.std(dim=0)
        logger.info(f'{sd.mean()=}')
        logger.info(f'{sd.std()=}')
        self.mean = mean
        self.std = std

        self.y = torch.tensor(self.img_hdf5['labels'][:load_num])

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.path, 'r')

    def __getitem__(self, idx):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        y = self.y[idx]
        x = self.x[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x.float(), y.int()

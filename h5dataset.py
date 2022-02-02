import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, path, load_num=100, transform=None,
                 target_transform=None):
        self.transform = transform
        self.load_num = load_num
        self.target_transform = target_transform
        self.path = path
        self.open_hdf5()
        self.x = torch.tensor(self.img_hdf5['imgs'][:load_num])
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

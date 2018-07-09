from torch.utils import data
import numpy as np
import torch
import h5py
from PIL import Image
from mpi4py import MPI


class Hdf5Dataset(data.Dataset):
    def __init__(self, file_path, transform=None, is_test=False):
        super(Hdf5Dataset, self).__init__()
        self.transform = transform
        hf = h5py.File(file_path)
        if not is_test:
            self.data = hf.get('X_train')
            self.target = hf.get('y_train')
        else:
            self.data = hf.get('X_test')
            self.target = hf.get('y_test')

    def __getitem__(self, index):
        img = np.uint8(self.data[index])
        # sample = Image.fromarray(img).convert('RGB')
        sample = Image.fromarray(img)
        sample.show()
        label = torch.from_numpy(np.asarray(self.target[index], dtype="uint8")).float()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return self.data.shape[0]


class Hdf5DatasetMPI(data.Dataset):
    def __init__(self, file_path, transform=None, is_test=False):
        super(Hdf5DatasetMPI, self).__init__()
        self.transform = transform
        hf = h5py.File(file_path, driver='mpio', comm=MPI.COMM_WORLD)
        if not is_test:
            self.data = hf.get('X_train')
            self.target = hf.get('y_train')
        else:
            self.data = hf.get('X_test')
            self.target = hf.get('y_test')

    def __getitem__(self, index):
        img = np.uint8(self.data[index])
        # sample = Image.fromarray(img).convert('RGB')
        sample = Image.fromarray(img)
        label = torch.from_numpy(np.asarray(self.target[index], dtype="uint8")).float()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return self.data.shape[0]

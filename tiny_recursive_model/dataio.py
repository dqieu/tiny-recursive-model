import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
import h5py

class HDF(Dataset):
    def __init__(self, path, excluded_idx, length = 255):
        self.base_ds = h5py.File(path, 'r')
        excluded_idx = set(excluded_idx)

        self.samples = []
        self.split_idx = {}

        for i in range(len(self.base_ds)):
            if i in excluded_idx:
                continue

            split = self.base_ds[str(i)].attrs['split']
            ds_idx = len(self.samples)

            self.samples.append(i)
            self.split_idx.setdefault(split, []).append(ds_idx)

        self.length = length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        real_idx = self.samples[idx]
        inp = self.base_ds[str(real_idx)]
        lab = inp.attrs['label']
        inp = inp[:]
        if inp.shape[0] > self.length:
            # Random crop
            start = random.randint(0, inp.shape[0] - self.length)
            inp = inp[start:start + self.length]
        # elif inp.shape[0] < self.length:
        #     # Pad
        #     pad_width = self.length - inp.shape[0]
        #     inp = np.pad(inp, ((0, pad_width), (0, 0)), mode='constant')
        return torch.Tensor(inp), lab

def padded_batch(batch):
    xs, ys = zip(*batch)
    # lengths = torch.tensor([x.shape[0] for x in xs])
    xs = pad_sequence(xs, batch_first=True)  # pad to max length
    # B, T, _ = xs.shape
    # pad_mask = torch.arange(T).expand(B, T) >= lengths
    ys = torch.tensor(ys)
    return xs, ys
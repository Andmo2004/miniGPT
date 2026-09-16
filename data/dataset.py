# data/dataset.py – PyTorch Dataset for language modelling

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

#
# TODO: Create a Dataset that serves (input, target) pairs of token sequences
#       for next-token prediction.
#
# ── How language model data works ──
#   Given a sequence of tokens [t0, t1, t2, t3, t4]:
#     input  (x) = [t0, t1, t2, t3]     ← the model sees this
#     target (y) = [t1, t2, t3, t4]     ← the model predicts this
#
#   The target is simply the input shifted by one position to the right.
#   At each position i, the model must predict y[i] given x[0:i+1].
#

# ── Class: TextDataset(torch.utils.data.Dataset) ──
class TextDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        
        """ Initialize TextDataset class """
        # TODO:
        #   1. Load the token IDs from the .npy file:
        #        (use int64 because PyTorch embedding layers expect LongTensor)
        #   2. Store block_size:
               
        self.data = np.load(data_path).astype(np.int64)
        self.block_size = block_size

    def __len__(self):
        """ Return the number of valid starting position """
        # TODO:
        #   Return the number of valid starting positions:
        #   We need at least block_size + 1 consecutive tokens to form
        #   one (x, y) pair, so the last valid start is len - block_size - 1.

        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        """ Grab a chunk of bs + 1 tokens starting at idx  """
        # TODO:
        #   1. Grab a chunk of block_size + 1 tokens starting at idx:
        #   2. Split into input and target:
        #   3. return x, y

        _chunk = self.data[idx : idx + self.block_size +1]
        x = torch.from_numpy(_chunk[: -1]) # first bs token
        y = torch.from_numpy(_chunk[1 : ]) # last bs token 

        return x, y
    
# ── Helper function: get_dataloaders ──
def get_dataloaders(data_dir, block_size, batch_size):
    """ Function to get data loaders for train and validation sets """
    # TODO:
    #   1. Create TextDataset for train and val:
    #   2. Wrap in DataLoader:
    #   3. return train_loader, val_loader
    #
    #   NOTE on shuffle: we shuffle train to avoid the model memorising
    #   the order of chunks.  Val is not shuffled because we just want a
    #   stable loss estimate.      
    train_ds = TextDataset(os.path.join(data_dir, "train.npy"), block_size)
    val_ds = TextDataset(os.path.join(data_dir, "val.npy"), block_size)

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

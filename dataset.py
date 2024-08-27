import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CacheDataset(Dataset):
    def __init__(self, trace_file):
        trace_vec = []
        with open('./cache_trace.log', 'r') as f:
            counter = 0
            while counter < 12800*10:
                line = f.readline()
                if not line:
                    break
                vec = line.strip().split()
                vec = [int(c) for c in vec]
                trace_vec.append(vec)
                counter += 1
        trace_vec = np.array(trace_vec, dtype=np.float32)
        self.trace_vec = trace_vec

    def __len__(self):
        return len(self.trace_vec) - 1

    def __getitem__(self, idx):
        return self.trace_vec[idx], self.trace_vec[idx+1]


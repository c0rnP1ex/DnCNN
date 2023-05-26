import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import os

class DenoisingDataset(Dataset):
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        clean = self.xs[index]
        noise = torch.randn(clean.size()).mul_(self.sigma/255.0)
        noisy = clean + noise
        return noisy, clean

    def __len__(self):
        return self.xs.size(0)
    
def datagenerator(data_dir="trainData/patches_40", verbose=False, batch_size = 32):
    file_names = os.listdir(data_dir)
    data = []
    for file_name in file_names:
        img = cv2.imread(data_dir + "/" + file_name, cv2.IMREAD_COLOR)
        h, w, c= img.shape
        data.append(img)
        if verbose:
            print(f'{file_name} / {str(len(file_names))} are loaded. {h}, {w}')
    data = np.array(data, dtype='uint8')
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('All data are loaded.')
    return data
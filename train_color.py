import os, datetime, time
import argparse
import numpy as np
import torch
import torch_npu
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import dataset_color as ds


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum')(input, target).div_(2)
    

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='trainData/patches_40', type=str, help='path of train data')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()



if __name__ == '__main__':
    device = torch.device('npu')
    model = DnCNN().to(device)
    criterion = sum_squared_error()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)

    n_epoch = args.epoch
    batch_size = args.batch_size
    save_dir = f'temp/model_sigma{args.sigma}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(0, n_epoch):
        xs = ds.datagenerator(data_dir=args.train_data, verbose=False, batch_size=batch_size)
        xs = xs.astype('float32')/255.0
        xs = torch.from_numpy(xs.transpose((0,3,1,2)))
        DDataset = ds.DenoisingDataset(xs, sigma=25)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        n = 0

        for n_count, batch_xy in enumerate(DLoader):
            model.train()
            optimizer.zero_grad()
            noisy, clean = batch_xy[0].to(device), batch_xy[1].to(device)
            loss = criterion(model(noisy), clean)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
            n = n_count
        elapsed_time = time.time() - start_time
        scheduler.step(epoch)

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n, elapsed_time))
        np.savetxt(f'train_result_sigma{args.sigma}.txt', np.hstack((epoch+1, epoch_loss/n, elapsed_time)), fmt='%2.4f')
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
import torch
import torch_npu
import cv2
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os

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

class denoise():

    def __init__(self, model, device, name_noisy_img) -> None:
        self.device = device
        self.model = model
        self.img = cv2.imread(name_noisy_img, cv2.IMREAD_COLOR)
        pass

    def main(self):
        data = []
        data.append(self.img)
        data = np.stack(data, axis=0)
        data = data.astype(np.float32)/255.0
        tensor = torch.from_numpy(data.transpose((0, 3, 1, 2))).contiguous().to(device)
        output = self.model(tensor)
        output = output.cpu().detach().numpy().transpose((0, 2, 3, 1))
        return output

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



if __name__ == '__main__':
    clean_dir = 'DataSets/CBSD68/original_png/'
    noisy_dir = 'DataSets/CBSD68/noisy25/'
    denoise_dir = 'output/denoise_sigma25_color_8000_40_17/'
    if not os.path.exists(denoise_dir):
        os.makedirs(denoise_dir)
    device = torch.device('npu')
    model = DnCNN().to(device)
    model = torch.load('temp/model_sigma25/model_100.pth')
    model.eval()
    file_names = os.listdir(clean_dir)
    # clean_imgs = []
    # noise_imgs = []
    # denoise_imgs = []
    n = 0
    psnr_sum_denoise = 0
    psnr_sum_noise = 0
    for file_name in file_names:
        clean_img = cv2.imread(clean_dir + "/" + file_name, cv2.IMREAD_COLOR)
        noise_img = cv2.imread(noisy_dir + "/" + file_name, cv2.IMREAD_COLOR)
        denoise_t = denoise(model, device, noisy_dir + "/" + file_name)
        denoise_img = denoise_t.main()[0, :, :, :]*255.0
        cv2.imwrite(f'{denoise_dir}{file_name}', denoise_img)
        # clean_imgs.append(clean_img)
        # noise_imgs.append(noise_img)
        # denoise_imgs.append(denoise_img)
        psnr_sum_denoise += psnr(clean_img, denoise_img)
        psnr_sum_noise += psnr(clean_img, noise_img)
        n += 1
    print(f'psnr: {psnr_sum_denoise/n}')
    print(f'psnr: {psnr_sum_noise/n}')
import cv2
import numpy as np
import os
import random

if __name__ == '__main__':
    dataset_path = "DataSets/DIV2K_train_HR"
    output_path = "trainData/patches_40"

    file_names = os.listdir(dataset_path)
    img = []
    total_num = 8000
    patch_size = 40
    step = 10
    count = 0
    for file_name in file_names:
        current_img = cv2.imread(dataset_path + "/" + file_name, cv2.IMREAD_COLOR)
        h0 = random.randint(0, 800)
        w0 = random.randint(0, 800)
        h, w, c = current_img.shape
        m = 2
        n = 5
        for j in range(0, m):
            for k in range(0, n):
                temp = current_img[h0+j*step+j*patch_size:h0+j*step+j*patch_size+patch_size, w0+k*step+k*patch_size:w0+k*step+k*patch_size+patch_size, :]
                count += 1
                cv2.imwrite(output_path + "/" + f'{count}.png', temp)
                print(f'{count}/{total_num}')
                
                    

import sys
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
import time






if __name__ == '__main__':
    

    # load cal psnr , ssim
    linear_mse=np.load('./psnr_ssim_results/linear_mse.npy')
    linear_psnr=np.load('./psnr_ssim_results/linear_psnr.npy')
    linear_ssim=np.load('./psnr_ssim_results/linear_ssim.npy')
    vfi_mse=np.load('./psnr_ssim_results/vfi_mse.npy')
    vfi_psnr=np.load('./psnr_ssim_results/vfi_psnr.npy')
    vfi_ssim=np.load('./psnr_ssim_results/vfi_ssim.npy')

    # shape (4, num of samples) , 4 -> x2, x4, x8, x16
    ratio = [2,4,8,16]

    vfi_mse_revised = []
    vfi_psnr_revised = []
    vfi_ssim_revised = []
    linear_mse_revised = []
    linear_psnr_revised = []
    linear_ssim_revised = []

    # extract generate frames
    for i,r in enumerate(ratio):
        vfi_mse_revised.append(vfi_mse[i][1::r])
        vfi_psnr_revised.append(vfi_psnr[i][1::r])
        vfi_ssim_revised.append(vfi_ssim[i][1::r])
        
        linear_mse_revised.append(linear_mse[i][1::r])
        linear_psnr_revised.append(linear_psnr[i][1::r])
        linear_ssim_revised.append(linear_ssim[i][1::r])
        
    # avg mse , ssim (don't cal psnr because of inf)
    for i,r in enumerate(ratio):
        print(f"x{r}")
        print(f"vfi -> mse : {vfi_mse[i].sum()/len(vfi_mse[i]):.3f} , ssim : {vfi_ssim[i].sum()/len(vfi_ssim[i]):.3f}")
        print(f"linear -> mse : {linear_mse[i].sum()/len(linear_mse[i]):.3f} , ssim : {linear_ssim[i].sum()/len(linear_ssim[i]):.3f}")
        

      
import sys
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import math
from tqdm import tqdm

import time
import argparse
import math
import torch
import cv2
import pdb
from dataloader_psnr_ssim import *
from metric_psnr_ssim import *

import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_data_root', default='/mnt/ICT_DATASET_FOLDS', type=str)
    parser.add_argument('--pred_data_root', default='/project/Project/ICT_medical_AD_VFI/ICT_MRI_interpolation/inference', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()

    ratios = ['x2','x4','x8','x16','x32']
    folds = ['0','1','2','3','4']
    methods = ['linear_results','laplacian_no_perceptual_results','laplacian_perceptual_results']
    
    
    _psnr = PSNR()
    _ssim = SSIM()
    
    
    df = pd.DataFrame(index = ratios)
    for method in methods:
        _pred_data_root = '/'.join([args.pred_data_root, method])
        for fold in folds:
            _ratio_psnr =[]
            _ratio_ssim =[]
            for ratio in ratios:    
                print(f"current work is method:{method} , fold:{fold} , ratio:{ratio}")
                
                dataset_val = PSNR_SSIM_NIfTIDataset(args.gt_data_root,_pred_data_root, batch_size=args.batch_size, fold=fold , ratio = ratio ,  mode='eval', method= method , args = args)
                _dataloader = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)
                
                _cur_psnr = 0
                _cur_ssim = 0
                for pred , gt in tqdm(_dataloader):
                    pred, gt = pred.cuda() , gt.cuda()
                    
                    _cur_psnr += _psnr(pred,gt)
                    _cur_ssim += _ssim(pred,gt)
                    
                _divider = int(ratio.strip("x"))
                #eliminate the number of gt frames (it adds _mean_evaltuon to zero) 
                _mean_psnr = _cur_psnr/ (len(_dataloader) - len(_dataloader)/  _divider)
                _mean_ssim = _cur_ssim/ (len(_dataloader) - len(_dataloader)/ _divider)
                
                _ratio_psnr.append(_mean_psnr.item()) 
                _ratio_ssim.append(_mean_ssim.item()) 
            
            
            df[f"{method}+'_fold_'{fold}'_'psnr'"] = _ratio_psnr 
            df[f"{method}+'_fold_'{fold}'_'ssim'"] = _ratio_ssim
            
                
                
                
    df.to_csv('./psnr_ssim_results.csv')
    print("results save done")
                    
                    
                
            
        
            
            
            
            
            
        
    
    
    
    
    
    
    
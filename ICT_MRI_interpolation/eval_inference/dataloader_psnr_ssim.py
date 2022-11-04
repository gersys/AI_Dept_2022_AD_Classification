import os
import cv2
import torch
import numpy as np
import random
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize
import pdb



class PSNR_SSIM_NIfTIDataset(Dataset):
    def __init__(self, gt_data_root, pred_data_root, exp=4, max_index=480, batch_size=16, fold = None, ratio=None, mode=None  , method = None, args= None):

        self.fold = fold
        assert isinstance(self.fold, str) , "undefined Dataset fold"
        
        self.ratio = ratio
        assert isinstance(self.ratio, str) , "undefined Dataset ratio"
        
        self.method = method
        assert isinstance(self.method, str) , "undefined Dataset ratio"

        self.mode = mode 
        assert self.mode == 'train' or self.mode == 'eval' , "undefined mode"
        
        

        self.args = args

        self.batch_size = batch_size
        self.h = 360
        self.w = 480
        self.exp = exp
        self.max_index = max_index
        self.gt_data_root = gt_data_root + f'/{self.fold}/{self.mode}'
        self.pred_data_root = pred_data_root + f'/{self.fold}/{self.mode}/{self.ratio}'
        
        
        self.gt_data_list = os.listdir(self.gt_data_root)
        self.gt_data_list[:] = [d for d in self.gt_data_list if self.is_data_valid(d)]
        self.pred_data_list = os.listdir(self.pred_data_root)
        self.pred_data_list[:] = [d for d in self.pred_data_list if self.is_data_valid(d)]
        

    def __len__(self):
        return len(self.gt_data_list * (self.max_index - 2))

    def is_data_valid(self, data_name, verbose=True):
        if len(glob(os.path.join(self.gt_data_root, data_name, '*.png'))) < self.max_index:
            # print('%s is not a valid data' % data_name)
            return False
        
        img = cv2.imread(os.path.join(self.gt_data_root, data_name, '0.png'))
        if img.shape[0] != self.h or img.shape[1] != self.w:
            # print('%s is not a valid data, shape: (%d, %d)' % (data_name, img.shape[0], img.shape[1]))
            return False
        
        if len(glob(os.path.join(self.pred_data_root, data_name, '*.png'))) < self.max_index:
            # print('%s is not a valid data' % data_name)
            return False
        
        img = cv2.imread(os.path.join(self.pred_data_root, data_name, '0.png'))
        if img.shape[0] != self.h or img.shape[1] != self.w:
            # print('%s is not a valid data, shape: (%d, %d)' % (data_name, img.shape[0], img.shape[1]))
            return False

        return True

    def getimg(self, index):
        dir_idx = index // (self.max_index - 2)
        img_idx = index % (self.max_index - 2) + 1

        pred_dir_path = os.path.join(self.pred_data_root, self.pred_data_list[dir_idx])
        gt_dir_path = os.path.join(self.gt_data_root, self.gt_data_list[dir_idx])
        
        pred = cv2.imread(os.path.join(pred_dir_path, '%d.png' % img_idx))
        gt = cv2.imread(os.path.join(gt_dir_path, '%d.png' % img_idx))
        return pred, gt

    def __getitem__(self, index):
        pred, gt = self.getimg(index)
        
        pred = torch.from_numpy(pred.copy()).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()
        
        return pred , gt
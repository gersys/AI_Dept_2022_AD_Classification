import os
import cv2
import torch
import numpy as np
import random
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Contrast():
    
    def __init__(self):
        intersection = 0.1617
        self.fn1_thresh_low = 0.12
        self.fn1_thresh_high = intersection
        self.fn1 = lambda x: 300 * (x - self.fn1_thresh_low) ** 2 + self.fn1_thresh_low
    
        self.fn2_thresh_low = intersection
        self.fn2_thresh_high = 0.20
        self.fn2 = lambda x: 300 * (x - self.fn2_thresh_high) ** 2 + self.fn2_thresh_high
    
    def forward(self, x):
        # x.shape = b, 3:3:3, h, w
        
        x = x[:, [0, 3, 6], ...]
        
        x_0_fn1 = torch.where((self.fn1_thresh_low <= x) * (x < self.fn1_thresh_high), self.fn1(x), x * 0)
        x_0_fn2 = torch.where((self.fn2_thresh_low <= x) * (x < self.fn2_thresh_high), self.fn2(x), x * 0)
        x_0_else = torch.where((x < self.fn1_thresh_low) + (self.fn2_thresh_high <= x), x, x * 0)
        
        x_0 = x_0_fn1 + x_0_fn2 + x_0_else
        x_1 = x
        x_2 = 2 * x_1 - x_0
        x = torch.cat([x_0, x_1, x_2], dim=-3)
        x = x[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], ...]
        
        return x
    
    def recon(self, x):
        return x.mean(dim=-3)



class NIfTIDataset(Dataset):
    def __init__(self, data_root, exp=4, max_index=430, batch_size=16, fold = None, mode=None , args= None):

        self.fold = fold
        assert isinstance(self.fold, int) , "undefined Dataset fold"

        self.mode = mode 
        assert self.mode == 'train' or self.mode == 'eval' , "undefined mode"

        self.args = args

        self.batch_size = batch_size
        self.h = 360
        self.w = 480
        self.exp = exp
        self.max_index = max_index
        self.data_root = data_root + f'/{self.fold}/{self.mode}'
        self.data_list = os.listdir(self.data_root)
        self.data_list[:] = [d for d in self.data_list if self.is_data_valid(d)]
        self.contrast = Contrast()

    def __len__(self):
        return len(self.data_list * (self.max_index - 2))

    def is_data_valid(self, data_name, verbose=True):
        if len(glob(os.path.join(self.data_root, data_name, '*.png'))) < self.max_index:
            # print('%s is not a valid data' % data_name)
            return False
        
        # import pdb
        # pdb.set_trace()
        
        img = cv2.imread(os.path.join(self.data_root, data_name, '0.png'))
        if img.shape[0] != self.h or img.shape[1] != self.w:
            # print('%s is not a valid data, shape: (%d, %d)' % (data_name, img.shape[0], img.shape[1]))
            return False

        return True

    def getimg(self, index):
        dir_idx = index // (self.max_index - 2)
        img_idx = index % (self.max_index - 2) + 1

        dir_path = os.path.join(self.data_root, self.data_list[dir_idx])
        intervals = []
        for i in range(self.exp):
            itv = 2**i
            if img_idx < itv or (self.max_index - img_idx) < itv or i>4:
                break
            else:
                intervals.append(itv)
        itv = random.choice(intervals)
        l_idx = img_idx - itv
        r_idx = img_idx + itv

        
        if self.args.model == 'FILM':
            img0 = cv2.imread(os.path.join(dir_path, '%d.png' % l_idx))[20:340,:,np.newaxis]
            gt = cv2.imread(os.path.join(dir_path, '%d.png' % img_idx))[20:340,:,np.newaxis]
            img1 = cv2.imread(os.path.join(dir_path, '%d.png' % r_idx))[20:340,:,np.newaxis]
        else:
            img0 = cv2.imread(os.path.join(dir_path, '%d.png' % l_idx))
            gt = cv2.imread(os.path.join(dir_path, '%d.png' % img_idx))
            img1 = cv2.imread(os.path.join(dir_path, '%d.png' % r_idx))
            

            
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)   # HWC to CHW
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        
        
        data_ = torch.cat((img0, img1, gt), 0)  # concat on channel axis
        data_ = resize(data_, [368, 480])  # image size must be in multiple of 16!
        
        return data_

        
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = 'vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
            

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.aug(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)

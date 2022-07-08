import os
import cv2
import torch
import numpy as np
import random
from glob import glob
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NIfTIDataset(Dataset):
    def __init__(self, data_root, exp=4, max_index=430, batch_size=16):
        self.batch_size = batch_size
        self.h = 360
        self.w = 480
        self.exp = exp
        self.max_index = max_index
        self.data_root = data_root
        self.data_list = os.listdir(data_root)
        self.data_list[:] = [d for d in self.data_list if self.is_data_valid(d)]

    def __len__(self):
        return len(self.data_list * (self.max_index - 2))

    def is_data_valid(self, data_name, verbose=True):
        if len(glob(os.path.join(self.data_root, data_name, '*.png'))) < self.max_index:
            print('%s is not a valid data' % data_name)
            return False
        
        img = cv2.imread(os.path.join(self.data_root, data_name, '0.png'))
        if img.shape[0] != self.h or img.shape[1] != self.w:
            print('%s is not a valid data, shape: (%d, %d)' % (data_name, img.shape[0], img.shape[1]))
            return False

        return True

    def getimg(self, index):
        dir_idx = index // (self.max_index - 2)
        img_idx = index % (self.max_index - 2) + 1

        dir_path = os.path.join(self.data_root, self.data_list[dir_idx])
        intervals = []
        for i in range(self.exp):
            itv = 2**i
            if img_idx < itv or (self.max_index - img_idx) < itv:
                break
            else:
                intervals.append(itv)
        itv = random.choice(intervals)
        l_idx = img_idx - itv
        r_idx = img_idx + itv

        img0 = cv2.imread(os.path.join(dir_path, '%d.png' % l_idx), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        gt = cv2.imread(os.path.join(dir_path, '%d.png' % img_idx), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        img1 = cv2.imread(os.path.join(dir_path, '%d.png' % r_idx), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)

        
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

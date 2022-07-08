import os
import shutil
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings




parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', required=True)
parser.add_argument('--exp', default=4, type=int)
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--gt_dir', default='/mnt/hdd0/ICT_DATASET_EVAL', type=str)
parser.add_argument('--out', default='output', type=str, help='output directory')

args = parser.parse_args()
INDEX_LIST = list(range(0, 480, 2**args.exp))


def linear_interpolation(img0 , img1):
    mid = 0.5 * (img0 + img1)
    return mid


for (root, dirs, files) in os.walk(args.img):


    # handle the case the number of z-axis is 360 
    if len(files) == 360:
        continue


    if not (len(files) >= len(INDEX_LIST) and files[0][-4:] == '.png'):
        continue

    outdir = os.path.join(args.out, os.path.basename(root))
    if os.path.exists(outdir):
        print(f'Warning: file name {root} is duplicated.')
    os.makedirs(outdir, exist_ok=True)

    error = 0
    count = 0
    for idx in range(len(INDEX_LIST) - 1):
        img0 = cv2.imread(os.path.join(root, str(INDEX_LIST[idx]) + '.png'), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        img1 = cv2.imread(os.path.join(root, str(INDEX_LIST[idx + 1]) + '.png'), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
        


        h, w, c = img0.shape
        
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        
        padding = ((0, ph - h), (0, pw - w),(0,0))
        
        img0 = np.pad(img0 , padding, 'constant' , constant_values=0)
        img1 = np.pad(img1 , padding, 'constant' , constant_values=0)


        img_list = [img0, img1]
        idx_list = [INDEX_LIST[idx], INDEX_LIST[idx + 1]]
        for i in range(args.exp):
            tmp = []
            tmp_idx = []
            for j in range(len(img_list) - 1):
                mid = linear_interpolation(img_list[j], img_list[j + 1])
                mid_idx = (idx_list[j] + idx_list[j + 1]) // 2
                tmp.append(img_list[j])
                tmp.append(mid)
                tmp_idx.append(idx_list[j])
                tmp_idx.append(mid_idx)
                count += 1
            tmp.append(img1)
            tmp_idx.append(INDEX_LIST[idx + 1])
            img_list = tmp
            idx_list = tmp_idx

        if not os.path.exists(args.out):
            os.mkdir(args.out)

        for i in range(len(img_list)):
            img = img_list[i][:h, :w]
            # gt_img = cv2.imread(os.path.join(root, str(idx_list[i]) + '.png'), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            # mse = np.sum((img - gt_img)**2) / (h * w)
            # error += mse
            # print('%s.png MSE : %.3f' % (idx_list[i], mse))
            cv2.imwrite(os.path.join(outdir, '{}.png'.format(idx_list[i])), img)
    
    if INDEX_LIST[-1] < 479:
        for i in range(INDEX_LIST[-1], 480):
            shutil.copy(os.path.join(args.gt_dir, os.path.basename(root), f'{i}.png'), os.path.join(outdir, f'{i}.png'))

        
    # print('Total MSE: %.3f' % (error / count))





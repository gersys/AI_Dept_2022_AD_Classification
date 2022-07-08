import os
import shutil
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', required=True)
parser.add_argument('--exp', default=4, type=int)
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--out', default='output', type=str, help='output directory')
parser.add_argument('--gt_dir', default='/mnt/hdd0/ICT_DATASET_EVAL', type=str)

args = parser.parse_args()


try:
    try:
        from model.RIFE_HDv2 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        from train_log.RIFE_HDv3 import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v3.x HD model.")
except:
    try:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
    except:
        from model.RIFE import Model
        model = Model(gray=True)
        model.load_model(args.modelDir, -1)
        print("Loaded custom Model")
model.eval()
model.device()


if os.path.exists(args.out):
    shutil.rmtree(args.out)
os.mkdir(args.out)

INDEX_LIST = list(range(0, 480, 2**args.exp))

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
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)


        img_list = [img0, img1]
        idx_list = [INDEX_LIST[idx], INDEX_LIST[idx + 1]]
        for i in range(args.exp):
            tmp = []
            tmp_idx = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
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
            img = (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            # gt_img = cv2.imread(os.path.join(root, str(idx_list[i]) + '.png'), cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]
            # mse = np.sum((img - gt_img)**2) / (h * w)
            # error += mse
            # print('%s.png MSE : %.3f' % (idx_list[i], mse))
            cv2.imwrite(os.path.join(outdir, '{}.png'.format(idx_list[i])), img)
    
    if INDEX_LIST[-1] < 479:
        for i in range(INDEX_LIST[-1], 480):
            shutil.copy(os.path.join(args.gt_dir, os.path.basename(root), f'{i}.png'), os.path.join(outdir, f'{i}.png'))

        
    # print('Total MSE: %.3f' % (error / count))


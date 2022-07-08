import os
import shutil
import nibabel as nib
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='directory of img files')
parser.add_argument('--gt', required=True, help='root directory of gt files')
parser.add_argument('--out', default='nii_output', type=str, help='output directory')
parser.add_argument('--len', default=480, type=int, help='')

args = parser.parse_args()

if os.path.exists(args.out):
    shutil.rmtree(args.out)
os.mkdir(args.out)

for (root, dirs, files) in os.walk(args.img):
    if not (len(files) >= args.len and files[0][-4:] == '.png'):
        continue

    gt_dir = os.path.join(args.gt, os.path.basename(root)) + '.nii'
    data = nib.load(gt_dir)
    x = data.get_fdata()
    max_val = np.max(x)
    
    for i in range(args.len):
        img = cv2.imread(os.path.join(root, f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        x[:, :, i] = img
    x = x / 255 * max_val
    new_data = nib.Nifti1Image(x, data.affine, data.header)
    nib.save(new_data, os.path.join(args.out, os.path.basename(gt_dir)))
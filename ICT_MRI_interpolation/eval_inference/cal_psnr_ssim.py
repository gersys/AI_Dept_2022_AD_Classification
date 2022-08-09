
import sys
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
import time
import argparse









def calculate_mse(img1, img2):
    mse = np.mean((img1 - img2)**2)
    return mse
    




def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--org_img_path', default='/mnt/ICT_DATASET_FOLDS', type=str)
    parser.add_argument('--vfi_img_path', default='/project/Project/ICT_medical_AD_VFI/ICT_MRI_interpolation/inference', type=str)
    parser.add_argument('--split', default=None, type=str)
    parser.add_argument('--perceptual', action= 'store_true' )
    parser.add_argument('--laplacian', action= 'store_true' )

    args = parser.parse_args()
    
    


    # img 이름 list loading
    _file_names = os.listdir(args.org_img_path)
    

    _z_axis = 480

    org_path ='/mnt/ICT_DATASET_EVAL/'
    
    if args.perceptual:
        if args.laplacian:
            vfi_path = args.vfi_img_path +'/laplacian_perceptual_results/'+f'{args.split}/'
        else:
            vfi_path = args.vfi_img_path +'/laplacian_no_perceptual_results/'+f'{args.split}/'
    else:
        if args.laplacian:
            vfi_path = args.vfi_img_path +'/laplacian_no_perceptual_results/'+f'{args.split}/'
        else:
            vfi_path = args.vfi_img_path +'/no_laplacian_no_perceptual_results/'+f'{args.split}/'
        
            
    linear_path = args.vfi_img_path +'/linear_results/'+f'{args.split}/'
    

    vfi_mse = [[],[],[],[]]
    vfi_psnr = [[],[],[],[]]
    vfi_ssim = [[],[],[],[]]
    linear_mse = [[],[],[],[]]
    linear_psnr = [[],[],[],[]]
    linear_ssim = [[],[],[],[]]

    save_path = './psnr_ssim_results/'
    
    ratios = ['x2','x4','x8','x16','x32']
    

    for _fn in tqdm(_file_names):
        cur_img_vfi_mse=[]
        cur_img_vfi_psnr=[]
        cur_img_vfi_ssim=[]
        cur_img_linear_mse=[]
        cur_img_linear_psnr=[]
        cur_img_linear_ssim=[]
        for i in range(_z_axis):
            i=str(i)
            
            # 원본 load

            a1 = time.time()
            org_img=cv2.imread(org_path+_fn+f'/{i}.png')
            a2 = time.time()

            if org_img.shape[1]!=_z_axis:
                break


            a3 = time.time()
            vfi_imgs=[]
            for ratio in ratios:
                vfi_imgs.append(cv2.imread(vfi_path+f'{ratio}/'+_fn+f'/{i}.png'))

            a4 = time.time()
            

            

            linear_imgs=[]
            # linear 생성본 load (x2 , x4 , x8 , x16)

            a5 = time.time()
            for ratio in ratios:
                linear_imgs.append(cv2.imread(linear_path+f'{ratio}/'+_fn+f'/{i}.png'))

            a6 = time.time()

            
            
            # #calulate vfi mse
            for i,vfi_img in enumerate(vfi_imgs):
                vfi_mse[i].append(calculate_mse(org_img, vfi_img))

            
            #calculate vfi ssim
            a9 = time.time()
            for i,vfi_img in enumerate(vfi_imgs):
                vfi_ssim[i].append(calculate_ssim(org_img, vfi_img))
            a10 = time.time()

            print(f"vfi ssim time {a10-a9:.3f}")


            #calulate linear mse

            for i,linear_img in enumerate(linear_imgs):
                linear_mse[i].append(calculate_mse(org_img, linear_img))

            
            #calculate linear ssim  
            a13 = time.time()
            
            for i,linear_img in enumerate(linear_imgs):
                linear_ssim[i].append(calculate_ssim(org_img, linear_img))
            a14 = time.time()

            print(f"linear ssim time {a14-a13:.3f}")
            
            # exit()
        

    save_path = './psnr_ssim_results/'
    os.makedirs(save_path, exist_ok=True)


    if args.perceptual:
        if args.laplacian:
            np.save(save_path+f"laplacian_perceptual_mse_{args.split}.npy",np.asarray(vfi_mse))
            np.save(save_path+f"laplacian_perceptual_ssim_{args.split}.npy",np.asarray(vfi_ssim))
        else:
            np.save(save_path+f"no_laplacian_perceptual_mse_{args.split}.npy",np.asarray(vfi_mse))
            np.save(save_path+f"no_laplacian_perceptual_ssim_{args.split}.npy",np.asarray(vfi_ssim))
    else:
        if args.laplacian:
            np.save(save_path+f"laplacian_no_perceptual_mse_{args.split}.npy",np.asarray(vfi_mse))
            np.save(save_path+f"laplacian_no_perceptual_ssim_{args.split}.npy",np.asarray(vfi_ssim))
        else:
            np.save(save_path+f"no_laplacian_no_perceptual_mse_{args.split}.npy",np.asarray(vfi_mse))
            np.save(save_path+f"no_laplacian_no_perceptual_ssim_{args.split}.npy",np.asarray(vfi_ssim))
            
                
    np.save(save_path+f"linear_mse_{args.split}.npy",np.asarray(linear_mse))
    np.save(save_path+f"linear_ssim_{args.split}.npy",np.asarray(linear_ssim))
    
    
    

    print("mse , psnr , ssim npy save done")





    # 원본과 VFI 와의 PSNR , SSIM 차이 계산 및 평균 분산구해서 출력

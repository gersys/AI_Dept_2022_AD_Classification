import nibabel as nib
import numpy as np
import os
import sys
import imageio
from tqdm import tqdm


SAVE_DIR = "/mnt/hdd0/ICT_DATASET_PNG"

def nii2png(filename, save_dir, axis=2):
    data = nib.load(filename).get_fdata()
    if data.shape[1] != 480 and data.shape[2] != 480:
        print('%s: ' % filename, data.shape, ' skipped')
        return
    # basename = '.'.join(os.path.basename(filename).split('.')[:-1])
    max_ = np.max(data)
    data = data / max_
    
    for i in range(data.shape[axis]):
        # img = np.take(data, i, axis=axis)
        img = data[:, :, i]
        img = np.uint8(img * 255)
        # savename = ''.join([basename, '_', str(i), '.png'])
        savename = ''.join([str(i), '.png'])
        imageio.imwrite(os.path.join(save_dir, savename), img)


if __name__ == "__main__":
    inputs = sys.argv[1:]

    os.makedirs(SAVE_DIR, exist_ok=True)
    for name in inputs:
        if not os.path.exists(name):
            print("%s not exists" % name)
            continue
        elif os.path.isdir(name):
            files = [os.path.join(name, f) for f in os.listdir(name) if os.path.isfile(os.path.join(name, f))]
            if len(files) == 764:
                files = files[678:]
            for f in tqdm(files):
                basename = '.'.join(os.path.basename(f).split('.')[:-1])
                save_path = os.path.join(SAVE_DIR, basename)
                # if os.path.exists(save_path):
                    # print(save_path, " already exists.")
                    # continue
                os.makedirs(save_path, exist_ok=True)
                nii2png(f, save_path)
        else:
            nii2png(name, SAVE_DIR)

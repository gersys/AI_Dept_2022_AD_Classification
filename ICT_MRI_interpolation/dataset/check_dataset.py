import os
import sys
import shutil
import csv
from turtle import pos
import pandas as pd
from tqdm import tqdm
import time








def num_of_label_check() -> list:
     
    print("check the number of pet_pos_ad & pet_neg_nc")

    print("data loading...")
    
    start_time = time.time()
    df=pd.read_excel("../labels/220420_th_with_demo.xlsx", engine='openpyxl')
    end_time = time.time()
    print(f"data loaded. the time spent is {end_time-start_time:.3f}s ")
    
    
    pos_ad=[]
    neg_nc=[]
    for code, pet , label in zip(tqdm(df['nuc_t1_inServer']),df["PET"],df['label']):
        if pet == 1 and label == "ADD":
            pos_ad.append(code.split(".")[0])
        elif pet == 0 and label == "NC":
            neg_nc.append(code.split(".")[0])
            

    print(f"total data number : {len(df['PET'])}")
    print(f"num of pet pos ad : {len(pos_ad)}")
    print(f"num of pet neg nc : {len(neg_nc)}")
    
    return pos_ad , neg_nc

def overlap_test(folds:list) -> None:
    #Check if there is any overlap
    
    for i in folds[0]:
        assert i not in fold[1] , "fold '0' sample overlap with fold[1]"
        assert i not in fold[2] , "fold '1' sample overlap with fold[1]"
        assert i not in fold[4] , "fold '2' sample overlap with fold[1]"
        assert i not in fold[3] , "fold '3' sample overlap with fold[1]"
    
    



def split_folds_for_cv(pos_ad:list , neg_nc:list, total_test_num:int = 130, split_num:int = 5)-> dict: 
    
    #divide test num into pos , neg halves
    each_test_num = total_test_num / 2
    
    folds = []
    
    #devide test data into 5 folds
    _start_idx = 0 
    for _split in range(split_num):
        _end_idx = _start_idx + each_test_num
        _cur_pos = pos_ad[_start_idx:_end_idx]
        _cur_neg = neg_nc[_start_idx:_end_idx]
        folds.append({"pos_ad": _cur_pos , "neg_nc":_cur_neg})
        _start_idx = _end_idx
    

    overlap_test(folds=folds)
        
    
        
     
    
        
    
        
        
        
        
    
    
    
    return folds
    
    
    

def aa():
    for i in pos_ad:
        try:
            trg_dir=i+"_resample"
            shutil.copytree("/mnt/hdd0/ICT_DATASET_ALL/"+trg_dir,"/mnt/hdd0/ICT_DATASET_EVAL/"+trg_dir)
            num_pos_ad+=1
        except:
            continue
            
        if num_pos_ad==50:
            break
        
    
    for i in neg_nc:
        try:
            trg_dir=i+"_resample"
            shutil.copytree("/mnt/hdd0/ICT_DATASET_ALL/"+trg_dir,"/mnt/hdd0/ICT_DATASET_EVAL/"+trg_dir)
            num_neg_nc+=1
        except:
            continue
        
        if num_neg_nc==50:
            break
    




# the code for dataset sanity check

if __name__=="__main__":
    num_of_label_check()
    
    
    
    pass
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
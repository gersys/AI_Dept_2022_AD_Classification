import os
import sys
import shutil
import csv
from turtle import pos
import pandas as pd
from tqdm import tqdm
import time








def num_of_label_check() -> List:
     
    print("check the number of pet_pos_ad & pet_neg_nc")

    print("data loading...")
    
    start_time = time.time()
    df=pd.read_excel("../labels/220420_th_with_demo.xlsx", engine='openpyxl')
    end_time = time.time()
    print(f"data loaded. the time spent is {end_time-start_time:.3f}s ")
    
    
    pos_add=[]
    neg_nc=[]
    for code, pet , label in zip(tqdm(df['nuc_t1_inServer']),df["PET"],df['label']):
        if pet == 1 and label == "ADD":
            pos_add.append(code.split(".")[0])
        elif pet == 0 and label == "NC":
            neg_nc.append(code.split(".")[0])
            

    print(f"total data number : {len(df['PET'])}")
    print(f"num of pet pos ad : {len(pos_add)}")
    print(f"num of pet neg nc : {len(neg_nc)}")
    
    return pos_add , neg_nc

def aa():
    for i in pos_add:
        try:
            trg_dir=i+"_resample"
            shutil.copytree("/mnt/hdd0/ICT_DATASET_ALL/"+trg_dir,"/mnt/hdd0/ICT_DATASET_EVAL/"+trg_dir)
            num_pos_add+=1
        except:
            continue
            
        if num_pos_add==50:
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
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
import os
import sys
import shutil
import csv
from turtle import pos
import pandas as pd








def num_of_label_check():
    df=pd.read_excel("../labels/220420_th_with_demo.xlsx", engine='openpyxl')
    
    pos_add=[]
    neg_nc=[]
    for code, pet , label in zip(df['nuc_t1_inServer'],df["PET"],df['label']):
        if pet == 1 and label == "ADD":
            pos_add.append(code.split(".")[0])
        elif pet == 0 and label == "NC":
            neg_nc.append(code.split(".")[0])
            
    
    num_pos_add=0
    num_neg_nc=0
    
    print(pos_add)
    print(pos_add.index("IDEA_052_120429_nuc_t1"))
    
    
    exit()
    
    while True:
        a=int(input("type index"))
        print(pos_add[a])
    
    pass


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
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
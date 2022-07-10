import os
import sys
import shutil
import csv
from turtle import pos
import pandas as pd
from tqdm import tqdm
import time
import random



class Split_Dataset:
    def __init__(self) -> None:

        self.pos_ad = []
        self.neg_ad = []
        self.folds = []
        

    def extract_pos_ad_neg_nc(self) -> None:
     
        print("check the number of pet_pos_ad & pet_neg_nc")

        print("data loading...")
        
        start_time = time.time()
        df=pd.read_excel("../labels/220420_th_with_demo.xlsx", engine='openpyxl')
        end_time = time.time()
        print(f"data loaded. the time spent is {end_time-start_time:.3f}s ")
        
        for code, pet , label in zip(tqdm(df['nuc_t1_inServer']),df["PET"],df['label']):
            if pet == 1 and label == "ADD":
                self.pos_ad.append(code.split(".")[0])
            elif pet == 0 and label == "NC":
                self.neg_nc.append(code.split(".")[0])
                

        print(f"total data number : {len(df['PET'])}")
        print(f"num of pet pos ad : {len(self.pos_ad)}")
        print(f"num of pet neg nc : {len(self.neg_nc)}")


        # shuffle pos_ad , neg_nc
        random.seed(9999)
        random.shuffle(self.pos_ad)
        random.shuffle(self.neg_nc)
        
        print("extract pos_ad , neg_nc done")


    def overlap_test(self) -> None:
        #Check if there is any overlap
        
        print("overlap test start")
        for i in tqdm(self.folds[0]["pos_ad"]):
            assert i not in self.folds[1]["pos_ad"] , "fold '0' pos_ad sample overlap with fold[1]"
            assert i not in self.folds[2]["pos_ad"] , "fold '0' pos_ad sample overlap with fold[2]"
            assert i not in self.folds[3]["pos_ad"] , "fold '0' pos_ad sample overlap with fold[3]"
            assert i not in self.folds[4]["pos_ad"] , "fold '0' pos_ad sample overlap with fold[4]"

        for i in tqdm(self.folds[0]["neg_nc"]):
            assert i not in self.folds[1]["neg_nc"] , "fold '0' neg_nc sample overlap with fold[1]"
            assert i not in self.folds[2]["neg_nc"] , "fold '0' neg_nc sample overlap with fold[2]"
            assert i not in self.folds[3]["neg_nc"] , "fold '0' neg_nc sample overlap with fold[3]"
            assert i not in self.folds[4]["neg_nc"] , "fold '0' neg_nc sample overlap with fold[4]"

        print("there is no overlap")
    
    
    def split_folds_for_cv(self, total_test_num:int = 130, split_num:int = 5)-> None: 
        
        #divide test num into pos , neg halves
        each_test_num = int(total_test_num / 2)
        
        #devide test data into 5 folds
        _start_idx = 0 
        for _split in range(split_num):
            _end_idx = _start_idx + each_test_num
            _cur_pos = self.pos_ad[_start_idx:_end_idx]
            _cur_neg = self.neg_nc[_start_idx:_end_idx]
            self.folds.append({"pos_ad": _cur_pos , "neg_nc":_cur_neg})
            _start_idx = _end_idx
        
        #test for there are overlap between folds
        overlap_test(folds=self.folds)
        
    
    def create_folds_dataset(folds: list) -> None:
        
        pass
        
    

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

    split_dataset=Split_Dataset()

    split_dataset.extract_pos_ad_neg_nc()
    split_dataset.split_folds_for_cv()
    


    
    
    
    pass
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
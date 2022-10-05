
import os
import sys

import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics





# Thickness Custom dataset

class Thickness_Dataset_interpolation(torch.utils.data.Dataset): 
    def __init__(self, mode='org', sub_mode=False):
        self.dataset = pd.read_excel('./220407_thickness_final.xlsx')
        self.eval_dataset = pd.read_excel('./ICT_EVAL_list.xlsx') 
        self.demo_dataset = pd.read_excel('./CHC_220420.xlsx')
        

        self.mode = mode
        
        self.test_data = []
        self.test_label = []
        
        self.train_names = os.listdir('/mnt/ICT_DATASET_PNG')
        self.test_names = os.listdir('/mnt/ICT_DATASET_EVAL')
        
        self._test=[]   

        self.sub_mode = sub_mode     



        # print(self.test_names)

        # test 데이터에 demopraphics 정보 모두 달려있는지 확인..하기 위한 용도임 -> 다달려 있음!!

        # self.train_names_check = [ '_'.join(i.split('_')[:3]) if i[0]=='A' else '_'.join(i.split('_')[:2])  for i in self.train_names]

        # self.test_names_check = [ '_'.join(i.split('_')[:3]) if i[0]=='A' else '_'.join(i.split('_')[:2])  for i in self.test_names]
        # # # print(self.test_names_check)
        # total = 0
        # temp = []
        # for code in self.demo_dataset['code']:
        #     temp.append(code)

        # for name in self.test_names_check:
        #     if name in temp:
        #         print(True)
        #         total+=1
        # print(total)

        # exit()

                    
        for test_name in self.test_names:
            cur_name= test_name.split('_')      
            cur_name= "_".join(cur_name[:-1])
            for name, Cingulate, Frontal, Parietal, Temporal, Occipital, label \
                in zip(self.dataset['nuc_t1_inServer'], self.dataset['Cingulate'], self.dataset['Frontal'], self.dataset['Parietal'], self.dataset['Temporal'], self.dataset['Occipital'] , self.dataset['Group']  ):
                name=name.split('.')[0]
                if cur_name == name:
                    if np.isnan(Cingulate or Frontal or Parietal or Temporal or Occipital ):
                        break 
                    else:
                        self.test_data.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                        self._test.append(name)
                    if label == 'ADD':
                            self.test_label.append([1])
                            break
                    elif label == 'NC':
                            self.test_label.append([0])
                            break
                    else:
                            print("label error")
                            raise RuntimeError
        
        
        self.eval_names_org = []
        self.eval_names_x2 = []
        self.eval_names_x4 = []
        self.eval_names_x8 = []
        self.eval_names_x16 = []
        
        self.eval_data_x2=[]
        self.eval_data_x4=[]
        self.eval_data_x8=[]
        self.eval_data_x16=[]
        
        self.eval_label_x2=[]
        self.eval_label_x4=[]
        self.eval_label_x8=[]
        self.eval_label_x16=[]
        
        for name, Cingulate, Frontal, Parietal, Temporal, Occipital \
            in zip(self.eval_dataset['Cth'], self.eval_dataset['Cingulate'], self.eval_dataset['Frontal'], self.eval_dataset['Parietal'], self.eval_dataset['Temporal'], self.eval_dataset['Occipital']  ):
            
            if pd.isnull(name):
                # print("name is null continue to next name")
                continue
            if "left.txt" in name:
                if "x2" in name:
                    self.eval_names_x2.append(name)
                    self.eval_data_x2.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x2.append(test_label)
                elif "x4" in name:
                    self.eval_names_x4.append(name)
                    self.eval_data_x4.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x4.append(test_label)
                elif "x8" in name:
                    self.eval_names_x8.append(name)
                    self.eval_data_x8.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x8.append(test_label)
                elif "x16" in name:
                    self.eval_names_x16.append(name)
                    self.eval_data_x16.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x16.append(test_label)
                else:
                    print("There is no matching name (x2, x4, x8 , x16")
                    raise RuntimeError
                
                     
            else:
                continue

        _mean = np.asarray([3.14755793, 3.08414156, 2.98152454, 3.1833588,  2.89523807])
        _std = np.asarray ([0.19398837, 0.15084428 ,0.18888524, 0.18555396, 0.19826587])




        # test data 중에서 x2 , x4 ,x8 ,x16 test data 있는것만 골라서 성능 비교하기 위함..
        self.test_data_x2 =[]
        self.test_data_x4 =[]
        self.test_data_x8 =[]
        self.test_data_x16 =[]

        self.test_label_x2 =[]
        self.test_label_x4 =[]
        self.test_label_x8 =[]
        self.test_label_x16 =[]

        temp_eval_names_x2 = [ '_'.join(i.split('_')[:6]) if i[0] == 'A' else '_'.join(i.split('_')[:5])  for i in self.eval_names_x2]
        temp_eval_names_x4 = [ '_'.join(i.split('_')[:6]) if i[0] == 'A' else '_'.join(i.split('_')[:5])  for i in self.eval_names_x4]
        temp_eval_names_x8 = [ '_'.join(i.split('_')[:6]) if i[0] == 'A' else '_'.join(i.split('_')[:5])  for i in self.eval_names_x8]
        temp_eval_names_x16 = [ '_'.join(i.split('_')[:6]) if i[0] == 'A' else '_'.join(i.split('_')[:5])  for i in self.eval_names_x16]

        for _name , _data , _label in zip(self.test_names , self.test_data , self.test_label):
            _name_temp=_name.split("_")
            _name_temp.remove('t1')
            _name = '_'.join(_name_temp)
            
            if _name in temp_eval_names_x2:
                self.test_data_x2.append(_data)
                self.test_label_x2.append(_label)
            if _name in temp_eval_names_x4:
                self.test_data_x4.append(_data)
                self.test_label_x4.append(_label)
            if _name in temp_eval_names_x8:
                self.test_data_x8.append(_data)
                self.test_label_x8.append(_label)
            if _name in temp_eval_names_x16:
                self.test_data_x16.append(_data)
                self.test_label_x16.append(_label)



        self.test_data = np.asarray(self.test_data)
        self.test_data = (self.test_data-_mean) / _std

        self.test_data_x2 = np.asarray(self.test_data_x2)
        self.test_data_x2 = (self.test_data_x2-_mean) / _std

        self.test_data_x4 = np.asarray(self.test_data_x4)
        self.test_data_x4 = (self.test_data_x4-_mean) / _std

        self.test_data_x8 = np.asarray(self.test_data_x8)
        self.test_data_x8 = (self.test_data_x8-_mean) / _std

        self.test_data_x16 = np.asarray(self.test_data_x16)
        self.test_data_x16 = (self.test_data_x16-_mean) / _std




        self.eval_data_x2 = np.asarray(self.eval_data_x2)
        self.eval_data_x2 = (self.eval_data_x2-_mean) / _std

        self.eval_data_x4 = np.asarray(self.eval_data_x4)
        self.eval_data_x4 = (self.eval_data_x4-_mean) / _std

        self.eval_data_x8 = np.asarray(self.eval_data_x8)
        self.eval_data_x8 = (self.eval_data_x8-_mean) / _std

        self.eval_data_x16 = np.asarray(self.eval_data_x16)
        self.eval_data_x16 = (self.eval_data_x16-_mean) / _std




    

    def __len__(self):
        if self.mode =='org':
            if self.sub_mode == False:
                return len(self.test_data)
            elif self.sub_mode == 'x2':
                return len(self.test_data_x2)
            elif self.sub_mode == 'x4':
                return len(self.test_data_x4)
            elif self.sub_mode == 'x8':
                return len(self.test_data_x8)
            elif self.sub_mode == 'x16':
                return len(self.test_data_x16)

        elif self.mode =='x2':
            return len(self.eval_data_x2)
        elif self.mode =='x4':
            return len(self.eval_data_x4)
        elif self.mode =='x8':
            return len(self.eval_data_x8)
        elif self.mode =='x16':
            return len(self.eval_data_x16)
        else:
            print("mode error you should choice between [org, x2, x4, x8, x16]")
            raise RuntimeError
        
            
            
            

    def __getitem__(self, idx):
        if self.mode == 'org':
            if self.sub_mode == False:
                x = torch.FloatTensor(self.test_data[idx])
                y = torch.FloatTensor(self.test_label[idx])
            elif self.sub_mode == 'x2':
                x = torch.FloatTensor(self.test_data_x2[idx])
                y = torch.FloatTensor(self.test_label_x2[idx])
            elif self.sub_mode == 'x4':
                x = torch.FloatTensor(self.test_data_x4[idx])
                y = torch.FloatTensor(self.test_label_x4[idx])
            elif self.sub_mode == 'x8':
                x = torch.FloatTensor(self.test_data_x8[idx])
                y = torch.FloatTensor(self.test_label_x8[idx])
            elif self.sub_mode == 'x16':
                x = torch.FloatTensor(self.test_data_x16[idx])
                y = torch.FloatTensor(self.test_label_x16[idx])

        elif self.mode =='x2':
            x = torch.FloatTensor(self.eval_data_x2[idx])
            y = torch.FloatTensor(self.eval_label_x2[idx])
        elif self.mode == 'x4':
            x = torch.FloatTensor(self.eval_data_x4[idx])
            y = torch.FloatTensor(self.eval_label_x4[idx])
        elif self.mode == 'x8':
            x = torch.FloatTensor(self.eval_data_x8[idx])
            y = torch.FloatTensor(self.eval_label_x8[idx])
        elif self.mode == 'x16':
            x = torch.FloatTensor(self.eval_data_x16[idx])
            y = torch.FloatTensor(self.eval_label_x16[idx])
        else:
            print("mode error you should choice between [org, x2, x4, x8, x16]")
            raise RuntimeError
        
            
        return x,y 
         


class Thickness_Dataset_interpolation_demo(torch.utils.data.Dataset): 
    def __init__(self, mode='org', sub_mode=False):
        self.dataset = pd.read_excel('./220407_thickness_final.xlsx')
        self.eval_dataset = pd.read_excel('./ICT_EVAL_list.xlsx') 
        self.demo_dataset = pd.read_excel('./CHC_220420.xlsx')
        

        self.mode = mode
        self.test_data_for_demo =[]
        self.test_demo_for_demo =[]
        self.test_label_for_demo =[]
        self.test_name_for_demo =[]
      
        self.train_names = os.listdir('/mnt/ICT_DATASET_PNG')
        self.test_names = os.listdir('/mnt/ICT_DATASET_EVAL')
        
        self._test=[]   

        self.sub_mode = sub_mode     



        # print(self.test_names)

        # test 데이터에 demopraphics 정보 모두 달려있는지 확인..하기 위한 용도임 -> 다달려 있음!!

        # self.train_names_check = [ '_'.join(i.split('_')[:3]) if i[0]=='A' else '_'.join(i.split('_')[:2])  for i in self.train_names]

        # self.test_names_check = [ '_'.join(i.split('_')[:3]) if i[0]=='A' else '_'.join(i.split('_')[:2])  for i in self.test_names]
        # # print(self.test_names_check)
        # total = 0
        # temp = []
        # for code in self.demo_dataset['code']:
        #     temp.append(code)

        # for name in self.test_names_check:
        #     if name in temp:
        #         print(True)
        #         total+=1

        # print(total)
        # exit()


        # exit()

                    
        for test_name in self.test_names:
            if test_name[0]=='A':
                  cur_name='_'.join(test_name.split('_')[:3])  
            else:
                  cur_name='_'.join(test_name.split('_')[:2])
            for name, num ,Cingulate, Frontal, Parietal, Temporal, Occipital, label , sex , age_pet , edu, pet, apoe_e4  \
          in zip(self.demo_dataset['code'],self.demo_dataset['num'] ,self.demo_dataset['Cingulate'], self.demo_dataset['Frontal'], self.demo_dataset['Parietal'], self.demo_dataset['Temporal'], self.demo_dataset['Occipital'] , self.demo_dataset['label'], self.demo_dataset['sex'], self.demo_dataset['age_pet'] ,self.demo_dataset['edu'], self.demo_dataset['PET'] , self.demo_dataset['APOE_e4']  ):
                  name=name.split('.')[0]
                  if cur_name == name:   
                        if Cingulate == -1 or Frontal == - 1 or Parietal == - 1 or Temporal == - 1 or Occipital == - 1 or sex == - 1 or age_pet == - 1 or edu == - 1 or pet == - 1 or apoe_e4 == -1:
                              break
                        else: 
                              if (sex=='M'):
                                    sex=0
                              else:
                                    sex=1
                              
                              self.test_data_for_demo.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                            #   self.test_demo_for_demo.append(np.array([sex, age_pet, edu, pet, apoe_e4]))
                            #   self.test_demo_for_demo.append(np.array([sex, age_pet, edu]))
                              self.test_demo_for_demo.append(np.array([sex, age_pet, edu, apoe_e4]))
                            #   self.test_demo_for_demo.append(np.array([apoe_e4]))
                              self.test_name_for_demo.append(name)
                        if label == 'ADD':
                              self.test_label_for_demo.append([1])
                              break
                        elif label == 'NC':
                              self.test_label_for_demo.append([0])
                              break
                        else:
                              print("label error")
                              raise RuntimeError
        
        
        self.eval_names_org = []
        self.eval_names_x2 = []
        self.eval_names_x4 = []
        self.eval_names_x8 = []
        self.eval_names_x16 = []
        
        self.eval_data_x2=[]
        self.eval_data_x4=[]
        self.eval_data_x8=[]
        self.eval_data_x16=[]


        self.eval_demo_x2=[]
        self.eval_demo_x4=[]
        self.eval_demo_x8=[]
        self.eval_demo_x16=[]
        
        self.eval_label_x2=[]
        self.eval_label_x4=[]
        self.eval_label_x8=[]
        self.eval_label_x16=[]
        
        for name, Cingulate, Frontal, Parietal, Temporal, Occipital \
            in zip(self.eval_dataset['Cth'], self.eval_dataset['Cingulate'], self.eval_dataset['Frontal'], self.eval_dataset['Parietal'], self.eval_dataset['Temporal'], self.eval_dataset['Occipital']  ):
            
            if pd.isnull(name):
                # print("name is null continue to next name")
                continue
            if "left.txt" in name:
                if "x2" in name:
                    self.eval_names_x2.append(name)
                    self.eval_data_x2.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label_for_demo):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x2.append(test_label)
                elif "x4" in name:
                    self.eval_names_x4.append(name)
                    self.eval_data_x4.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label_for_demo):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x4.append(test_label)
                elif "x8" in name:
                    self.eval_names_x8.append(name)
                    self.eval_data_x8.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label_for_demo):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x8.append(test_label)
                elif "x16" in name:
                    self.eval_names_x16.append(name)
                    self.eval_data_x16.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                    for test_name, test_label in zip(self.test_names , self.test_label_for_demo):
                        cur_eval_name = "_".join(name.split("_")[:3])
                        cur_test_name ="_".join(test_name.split("_")[:3])
                        if cur_eval_name == cur_test_name:
                            self.eval_label_x16.append(test_label)
                else:
                    print("There is no matching name (x2, x4, x8 , x16")
                    raise RuntimeError
                
                     
            else:
                continue



        # x2, x4, x8 ,x16 data에 demo data 붙여 주는 작업


        temp_eval_names_x2_for_demo = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x2]
        temp_eval_names_x4_for_demo = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x4]
        temp_eval_names_x8_for_demo = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x8]
        temp_eval_names_x16_for_demo = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x16]

        for i in temp_eval_names_x2_for_demo:
            _idx=self.test_name_for_demo.index(i)            
            self.eval_demo_x2.append(self.test_demo_for_demo[_idx])

        for i in temp_eval_names_x4_for_demo:
            _idx=self.test_name_for_demo.index(i)            
            self.eval_demo_x4.append(self.test_demo_for_demo[_idx])

        for i in temp_eval_names_x8_for_demo:
            _idx=self.test_name_for_demo.index(i)            
            self.eval_demo_x8.append(self.test_demo_for_demo[_idx])

        for i in temp_eval_names_x16_for_demo:
            _idx=self.test_name_for_demo.index(i)            
            self.eval_demo_x16.append(self.test_demo_for_demo[_idx])

        

        _mean_demo = np.asarray([3.14632784 ,3.08416757, 2.98115842 ,3.18269986 ,2.89080612])
        _std_demo = np.asarray ([0.1949455  ,0.14873358, 0.18985266 ,0.18376173, 0.19961221])

        _age_pet_mean_demo = 69.50875486381322
        _edu_mean_demo = 11.809824902723735

        _age_pet_std_demo = 9.3678044347774
        _edu_std_demo = 4.777955903444009



        self.test_data_for_demo =  np.asarray(self.test_data_for_demo)
        self.test_data_for_demo = (self.test_data_for_demo-_mean_demo) / _std_demo

        self.test_demo_for_demo =  np.asarray(self.test_demo_for_demo)

        self.test_demo_for_demo[:,1] = (self.test_demo_for_demo[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        self.test_demo_for_demo[:,2] = (self.test_demo_for_demo[:,2]-_edu_mean_demo)/_edu_std_demo


        self.eval_data_x2 = np.asarray(self.eval_data_x2)
        self.eval_data_x2 = (self.eval_data_x2-_mean_demo) / _std_demo

        self.eval_data_x4 = np.asarray(self.eval_data_x4)
        self.eval_data_x4 = (self.eval_data_x4-_mean_demo) / _std_demo

        self.eval_data_x8 = np.asarray(self.eval_data_x8)
        self.eval_data_x8 = (self.eval_data_x8-_mean_demo) / _std_demo

        self.eval_data_x16 = np.asarray(self.eval_data_x16)
        self.eval_data_x16 = (self.eval_data_x16-_mean_demo) / _std_demo

        self.eval_demo_x2 = np.asarray(self.eval_demo_x2)
        self.eval_demo_x2[:,1] = (self.eval_demo_x2[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        self.eval_demo_x2[:,2] = (self.eval_demo_x2[:,2]-_edu_mean_demo)/_edu_std_demo

        self.eval_demo_x4 = np.asarray(self.eval_demo_x4)
        self.eval_demo_x4[:,1] = (self.eval_demo_x4[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        self.eval_demo_x4[:,2] = (self.eval_demo_x4[:,2]-_edu_mean_demo)/_edu_std_demo

        self.eval_demo_x8 = np.asarray(self.eval_demo_x8)
        self.eval_demo_x8[:,1] = (self.eval_demo_x8[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        self.eval_demo_x8[:,2] = (self.eval_demo_x8[:,2]-_edu_mean_demo)/_edu_std_demo


        self.eval_demo_x16 = np.asarray(self.eval_demo_x16)
        self.eval_demo_x16[:,1] = (self.eval_demo_x16[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        self.eval_demo_x16[:,2] = (self.eval_demo_x16[:,2]-_edu_mean_demo)/_edu_std_demo





        # test data 중에서 x2 , x4 ,x8 ,x16 test data 있는것만 골라서 성능 비교하기 위함..
        
        
        self.test_name_x2 =[]
        self.test_name_x4 =[]
        self.test_name_x8 =[]
        self.test_name_x16 =[]

        
        
        self.test_data_x2 =[]
        self.test_data_x4 =[]
        self.test_data_x8 =[]
        self.test_data_x16 =[]

        self.test_demo_x2 =[]
        self.test_demo_x4 =[]
        self.test_demo_x8 =[]
        self.test_demo_x16 =[]

        self.test_label_x2 =[]
        self.test_label_x4 =[]
        self.test_label_x8 =[]
        self.test_label_x16 =[]

        temp_eval_names_x2 = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x2]
        temp_eval_names_x4 = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x4]
        temp_eval_names_x8 = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x8]
        temp_eval_names_x16 = [ '_'.join(i.split('_')[:3]) if i[0] == 'A' else '_'.join(i.split('_')[:2])  for i in self.eval_names_x16]

        self.eval_names_x2 = temp_eval_names_x2
        self.eval_names_x4 = temp_eval_names_x4
        self.eval_names_x8 = temp_eval_names_x8
        self.eval_names_x16 = temp_eval_names_x16 



        for _name , _data , _label, _demo in zip(self.test_name_for_demo , self.test_data_for_demo , self.test_label_for_demo, self.test_demo_for_demo):

            # _name_temp=_name.split("_")
            # _name_temp.remove('t1')
            # _name = '_'.join(_name_temp)

            _data , _demo = np.asarray(_data) , np.asarray(_demo)

            if _name in temp_eval_names_x2:
                self.test_name_x2.append(_name)
                self.test_data_x2.append(_data)
                self.test_label_x2.append(_label)
                self.test_demo_x2.append(_demo)
            if _name in temp_eval_names_x4:
                self.test_name_x4.append(_name)
                self.test_data_x4.append(_data)
                self.test_label_x4.append(_label)
                self.test_demo_x4.append(_demo)
            if _name in temp_eval_names_x8:
                self.test_name_x8.append(_name)
                self.test_data_x8.append(_data)
                self.test_label_x8.append(_label)
                self.test_demo_x8.append(_demo)
            if _name in temp_eval_names_x16:
                self.test_name_x16.append(_name)
                self.test_data_x16.append(_data)
                self.test_label_x16.append(_label)
                self.test_demo_x16.append(_demo)


        


    

    def __len__(self):
        if self.mode =='org':
            if self.sub_mode == False:
                return len(self.test_data_for_demo)
            elif self.sub_mode == 'x2':
                return len(self.test_data_x2)
            elif self.sub_mode == 'x4':
                return len(self.test_data_x4)
            elif self.sub_mode == 'x8':
                return len(self.test_data_x8)
            elif self.sub_mode == 'x16':
                return len(self.test_data_x16)

        elif self.mode =='x2':
            return len(self.eval_data_x2)
        elif self.mode =='x4':
            return len(self.eval_data_x4)
        elif self.mode =='x8':
            return len(self.eval_data_x8)
        elif self.mode =='x16':
            return len(self.eval_data_x16)
        else:
            print("mode error you should choice between [org, x2, x4, x8, x16]")
            raise RuntimeError
        
            
            
            

    def __getitem__(self, idx):
        if self.mode == 'org':
            if self.sub_mode == False:
                x = torch.FloatTensor(self.test_data_for_demo[idx])
                y = torch.FloatTensor(self.test_label_for_demo[idx])
                d = torch.FloatTensor(self.test_demo_for_demo[idx])

            elif self.sub_mode == 'x2':
                x = torch.FloatTensor(self.test_data_x2[idx])
                y = torch.FloatTensor(self.test_label_x2[idx])
                d = torch.FloatTensor(self.test_demo_x2[idx])
            elif self.sub_mode == 'x4':
                x = torch.FloatTensor(self.test_data_x4[idx])
                y = torch.FloatTensor(self.test_label_x4[idx])
                d = torch.FloatTensor(self.test_demo_x4[idx])
            elif self.sub_mode == 'x8':
                x = torch.FloatTensor(self.test_data_x8[idx])
                y = torch.FloatTensor(self.test_label_x8[idx])
                d = torch.FloatTensor(self.test_demo_x8[idx])
            elif self.sub_mode == 'x16':
                x = torch.FloatTensor(self.test_data_x16[idx])
                y = torch.FloatTensor(self.test_label_x16[idx])
                d = torch.FloatTensor(self.test_demo_x16[idx])

        elif self.mode =='x2':
            x = torch.FloatTensor(self.eval_data_x2[idx])
            y = torch.FloatTensor(self.eval_label_x2[idx])
            d = torch.FloatTensor(self.eval_demo_x2[idx])
        elif self.mode == 'x4':
            x = torch.FloatTensor(self.eval_data_x4[idx])
            y = torch.FloatTensor(self.eval_label_x4[idx])
            d = torch.FloatTensor(self.eval_demo_x4[idx])
        elif self.mode == 'x8':
            x = torch.FloatTensor(self.eval_data_x8[idx])
            y = torch.FloatTensor(self.eval_label_x8[idx])
            d = torch.FloatTensor(self.eval_demo_x8[idx])
        elif self.mode == 'x16':
            x = torch.FloatTensor(self.eval_data_x16[idx])
            y = torch.FloatTensor(self.eval_label_x16[idx])
            d = torch.FloatTensor(self.eval_demo_x16[idx])
        else:
            print("mode error you should choice between [org, x2, x4, x8, x16]")
            raise RuntimeError
        
            
        return x,y,d




class Linear_Model(torch.nn.Module):
      
      def __init__(self):
            super(Linear_Model, self).__init__()
            self.linear1 = torch.nn.Linear(5,100, bias=False)
            self.bn1 = torch.nn.BatchNorm1d(100)
            self.linear2 = torch.nn.Linear(100,100, bias=False)
            self.bn2 = torch.nn.BatchNorm1d(100)
            self.linear3 = torch.nn.Linear(100,2, bias=False)
            
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

            for mod in self.modules():
                  if isinstance(mod, nn.Linear):
                        mod.weight.data.normal_(mean=0.0, std=1.)
                        if mod.bias is not None:
                              mod.bias.data.zero_()
      
      def forward(self, x):
            x = self.linear1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.linear3(x)
            return x



class Linear_Model_demo(torch.nn.Module):
      
      def __init__(self):
            super(Linear_Model_demo, self).__init__()
            self.linear1 = torch.nn.Linear(10,100, bias=False)
            self.linear1_1 = torch.nn.Linear(8,100, bias=False)
            self.linear1_2 = torch.nn.Linear(9,100, bias=False)
            self.linear1_3 = torch.nn.Linear(6,100, bias=False)
            self.bn1 = torch.nn.BatchNorm1d(100)
            self.linear2 = torch.nn.Linear(100,100, bias=False)
            self.bn2 = torch.nn.BatchNorm1d(100)
            self.linear3 = torch.nn.Linear(100,2, bias=False)
            
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()

            for mod in self.modules():
                  if isinstance(mod, nn.Linear):
                        mod.weight.data.normal_(mean=0.0, std=1.)
                        if mod.bias is not None:
                              mod.bias.data.zero_()
      
      def forward(self, x):
            x = self.linear1_2(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.linear3(x)
            return x


def plot_roc_curve(fper, tper,mode):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    os.makedirs("./roc_curve",exist_ok=True)
    plt.savefig(f'./roc_curve/{mode}.jpg')
    plt.clf()


def test(test_loader, model, mode):
    total = 0
    ans = 0
    
    
    all_scores = []
    all_preds=[]
    all_labels=[]

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # inputs , labels = inputs.cuda() , labels.cuda()
            outputs=model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            labels = torch.squeeze(labels.long())
            
            
            
            all_scores.extend(outputs[:,1].numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            


            ans+=(preds==labels).sum()
            total+=inputs.shape[0]
            

    all_scores = np.asarray(all_scores)
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    
    
    # print(all_scores)
    # print(all_preds)
    
    
    auc_score = roc_auc_score(all_labels, all_scores)
    
    fper, tper, thresholds = roc_curve(all_labels, all_scores)
    plot_roc_curve(fper, tper , mode)
    
    th_acc=[]
    
    for th in thresholds:
        all_preds = (all_scores >th)
        _cur_acc = (all_preds==all_labels).sum() / len(all_labels)
        
        th_acc.append(_cur_acc)
        
    th_acc = np.asarray(th_acc)
    max_th_acc = np.max(th_acc)
    
    

    cur_acc = ans/total 
      
    print(f"mode: {mode} acc: {cur_acc:.3f} auc_score: {auc_score:.3f}")        
    print(f"th max acc : {max_th_acc:.3f}")
    
      
    

def test_demo(test_loader, model, mode):
      
    total = 0
    ans = 0
    
    all_scores = []
    all_probs = []
    all_preds=[]
    all_labels=[]

    softmax = nn.Softmax(dim=1)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels , demos) in enumerate(test_loader):
                # inputs , labels, demos = inputs.cuda() , labels.cuda() , demos.cuda()
                # inputs = torch.cat((inputs, demos), dim=1)

                outputs=model(inputs)
                probs = softmax(outputs)
                
                preds = torch.argmax(outputs, dim=1)
                labels = torch.squeeze(labels.long())
                
                all_scores.extend(outputs[:,1].numpy())
                all_probs.extend(probs[:,1].numpy())
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())


                ans+=(preds==labels).sum()
                total+=inputs.shape[0]
            
    cur_acc = ans/total 
    
    auc_score = roc_auc_score(all_labels, all_scores)
    auc_by_prob = roc_auc_score(all_labels,all_probs)
    fper, tper, thresholds = roc_curve(all_labels, all_probs)
    
    th_acc=[]
    
    for th in thresholds:
        all_preds = (all_probs >th)
        _cur_acc = (all_preds==all_labels).sum() / len(all_labels)
        
        th_acc.append(_cur_acc)
        
    th_acc = np.asarray(th_acc)
    max_th_acc = np.max(th_acc)
    
      
    print(f"mode: {mode} acc: {cur_acc:.3f} auc_score: {auc_score:.3f} auc_by_probs: {auc_by_prob:.3f}")      
    print(f"th max acc : {max_th_acc:.3f}")  

    
    return (all_labels==all_preds) , all_labels

              
    


def esb_test(test_loader, models, mode):
      total = 0
      ans = 0
      
      model.eval()
      with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # inputs , labels = inputs.cuda() , labels.cuda()
            outputs_1=models[0](inputs)
            outputs_2=models[1](inputs)
            outputs_3=models[2](inputs)
            outputs_4=models[3](inputs)
            outputs_5=models[4](inputs)

            outputs =(outputs_1+outputs_2+outputs_3+outputs_4+outputs_5)/5

            preds = torch.argmax(outputs, dim=1)

            labels = torch.squeeze(labels.long())
            ans+=(preds==labels).sum()
            total+=inputs.shape[0]

            
              
              
      
      cur_acc = ans/total 
      
      
      print(f"esb mode: {mode} acc: {cur_acc:.2f}")  


def esb_test_demo(test_loader, models, mode):
      total = 0
      ans = 0
      
      model.eval()
      with torch.no_grad():
        for i, (inputs, labels , demos) in enumerate(test_loader):
                # inputs , labels, demos = inputs.cuda() , labels.cuda() , demos.cuda()
                inputs = torch.cat((inputs, demos), dim=1)
                outputs_1=models[0](inputs)
                outputs_2=models[1](inputs)
                outputs_3=models[2](inputs)
                outputs_4=models[3](inputs)
                outputs_5=models[4](inputs)
                
                outputs =(outputs_1+outputs_2+outputs_3+outputs_4+outputs_5)/5
                
                preds = torch.argmax(outputs, dim=1)
            
                labels = torch.squeeze(labels.long())
                ans+=(preds==labels).sum()
                total+=inputs.shape[0]
              
            
              
              
      
      cur_acc = ans/total 
      
      
      print(f"esb mode: {mode} acc: {cur_acc:.2f}")  
      
      




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    
    #load Dataset
    
    
    # print("setting eval datasets")
    # org_Dataset = Thickness_Dataset_interpolation(mode='org')
    # org_Dataset_x2 = Thickness_Dataset_interpolation(mode='org', sub_mode='x2')
    # org_Dataset_x4 = Thickness_Dataset_interpolation(mode='org', sub_mode='x4')
    # org_Dataset_x8 = Thickness_Dataset_interpolation(mode='org', sub_mode='x8')
    # org_Dataset_x16 = Thickness_Dataset_interpolation(mode='org', sub_mode='x16')





    # x2_Dataset = Thickness_Dataset_interpolation(mode='x2')
    # x4_Dataset = Thickness_Dataset_interpolation(mode='x4')
    # x8_Dataset = Thickness_Dataset_interpolation(mode='x8')
    # x16_Dataset = Thickness_Dataset_interpolation(mode='x16')
    
    
    
    
    # #define dataloader
    # org_loader = torch.utils.data.DataLoader(org_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x2 = torch.utils.data.DataLoader(org_Dataset_x2, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x4 = torch.utils.data.DataLoader(org_Dataset_x4, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x8 = torch.utils.data.DataLoader(org_Dataset_x8, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x16 = torch.utils.data.DataLoader(org_Dataset_x16, batch_size=32, shuffle=False, num_workers=8)

    # x2_loader = torch.utils.data.DataLoader(x2_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x4_loader = torch.utils.data.DataLoader(x4_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x8_loader = torch.utils.data.DataLoader(x8_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x16_loader = torch.utils.data.DataLoader(x16_Dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # #define model 
    # model1 = Linear_Model()
    # model2 = Linear_Model()
    # model3 = Linear_Model()
    # model4 = Linear_Model()
    # model5 = Linear_Model()
    
    # #load model saved model parameter
    # model1.load_state_dict(torch.load('./save_backup/epoch_345_acc_0.83.pth'))
    # model2.load_state_dict(torch.load('./save_backup/epoch_348_acc_0.83.pth'))
    # model3.load_state_dict(torch.load('./save_backup/epoch_351_acc_0.83.pth'))
    # model4.load_state_dict(torch.load('./save_backup/epoch_354_acc_0.83.pth'))
    # model5.load_state_dict(torch.load('./save_backup/epoch_366_acc_0.83.pth'))
    
    # models= [model1,model2,model3,model4,model5]
    # #test performance org , x2 , x4, x8,  x16
    # for i, model in enumerate(models):
    #     print(f"model {i}")    
    #     test(org_loader, model, 'org')
    #     test(org_loader_x2, model, 'org_x2')
    #     test(x2_loader, model, 'x2')
    #     test(org_loader_x4, model, 'org_x4')
    #     test(x4_loader, model, 'x4')
    #     test(org_loader_x8, model, 'org_x8')
    #     test(x8_loader, model, 'x8')
    #     test(org_loader_x16, model, 'org_x16')
    #     test(x16_loader, model, 'x16')
    #     print("")
        
        
        
    
    # print("")
    
    

    # esb_test(org_loader, models, 'org')
    # esb_test(org_loader_x2, models, 'org_x2')
    # esb_test(x2_loader, models, 'x2')
    # esb_test(org_loader_x4, models, 'org_x4')
    # esb_test(x4_loader, models, 'x4')
    # esb_test(org_loader_x8, models, 'org_x8')
    # esb_test(x8_loader, models, 'x8')
    # esb_test(org_loader_x16, models, 'org_x16')
    # esb_test(x16_loader, models, 'x16')


    #demo 포함한 test

    # org_Dataset = Thickness_Dataset_interpolation_demo(mode='org')
    # org_Dataset_x2 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x2')
    # org_Dataset_x4 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x4')
    # org_Dataset_x8 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x8')
    # org_Dataset_x16 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x16')


    # x2_Dataset = Thickness_Dataset_interpolation_demo(mode='x2')
    # x4_Dataset = Thickness_Dataset_interpolation_demo(mode='x4')
    # x8_Dataset = Thickness_Dataset_interpolation_demo(mode='x8')
    # x16_Dataset = Thickness_Dataset_interpolation_demo(mode='x16')

    # #define dataloader
    # org_loader = torch.utils.data.DataLoader(org_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x2 = torch.utils.data.DataLoader(org_Dataset_x2, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x4 = torch.utils.data.DataLoader(org_Dataset_x4, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x8 = torch.utils.data.DataLoader(org_Dataset_x8, batch_size=32, shuffle=False, num_workers=8)
    # org_loader_x16 = torch.utils.data.DataLoader(org_Dataset_x16, batch_size=32, shuffle=False, num_workers=8)

    # x2_loader = torch.utils.data.DataLoader(x2_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x4_loader = torch.utils.data.DataLoader(x4_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x8_loader = torch.utils.data.DataLoader(x8_Dataset, batch_size=32, shuffle=False, num_workers=8)
    # x16_loader = torch.utils.data.DataLoader(x16_Dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # #define model 
    # model1 = Linear_Model_demo()
    # model2 = Linear_Model_demo()
    # model3 = Linear_Model_demo()
    # model4 = Linear_Model_demo()
    # model5 = Linear_Model_demo()


    # #load model saved model parameter
    
    # # model1.load_state_dict(torch.load('./save_demo_3/epoch_314_acc_0.84.pth'))
    # # model2.load_state_dict(torch.load('./save_demo_3/epoch_359_acc_0.84.pth'))
    # # model3.load_state_dict(torch.load('./save_demo_3/epoch_379_acc_0.84.pth'))
    # # model4.load_state_dict(torch.load('./save_demo_3/epoch_575_acc_0.84.pth'))
    # # model5.load_state_dict(torch.load('./save_demo_3/epoch_592_acc_0.84.pth'))
    
    # model1.load_state_dict(torch.load('./save_demo_4/epoch_285_acc_0.87.pth'))
    # model2.load_state_dict(torch.load('./save_demo_4/epoch_459_acc_0.87.pth'))
    # model3.load_state_dict(torch.load('./save_demo_4/epoch_471_acc_0.88.pth'))
    # model4.load_state_dict(torch.load('./save_demo_4/epoch_478_acc_0.88.pth'))
    # model5.load_state_dict(torch.load('./save_demo_4/epoch_638_acc_0.88.pth'))
    
    # # model1.load_state_dict(torch.load('./save_demo_apoe/epoch_213_acc_0.88.pth'))
    # # model2.load_state_dict(torch.load('./save_demo_apoe/epoch_274_acc_0.88.pth'))
    # # model3.load_state_dict(torch.load('./save_demo_apoe/epoch_306_acc_0.88.pth'))
    # # model4.load_state_dict(torch.load('./save_demo_apoe/epoch_337_acc_0.88.pth'))
    # # model5.load_state_dict(torch.load('./save_demo_apoe/epoch_347_acc_0.89.pth'))
    
    # models= [model1,model2,model3,model4,model5]
    # #test performance org , x2 , x4, x8,  x16
    # for i, model in enumerate(models):
    #     print(f"model {i}")    
    #     org_conf_mat     ,  _= test_demo(org_loader, model, 'org')
    #     org_conf_mat_x2  , org_all_label_x2 = test_demo(org_loader_x2, model, 'org_x2')
    #     _conf_mat_x2     , _all_label_x2    = test_demo(x2_loader, model, 'x2')
    #     org_conf_mat_x4  , org_all_label_x4 = test_demo(org_loader_x4, model, 'org_x4')
    #     _conf_mat_x4     , _all_label_x4    = test_demo(x4_loader, model, 'x4')
    #     org_conf_mat_x8  , org_all_label_x8 = test_demo(org_loader_x8, model, 'org_x8')
    #     _conf_mat_x8     , _all_label_x8    = test_demo(x8_loader, model, 'x8')
    #     org_conf_mat_x16 , org_all_label_x16 = test_demo(org_loader_x16, model, 'org_x16')
    #     _conf_mat_x16    , _all_label_x16    = test_demo(x16_loader, model, 'x16')
        
        
        
    #     org_idx_x2=np.argsort(np.asarray(org_Dataset_x2.test_name_x2))
    #     idx_x2=np.argsort(np.asarray(x2_Dataset.eval_names_x2))
    #     org_idx_x4=np.argsort(np.asarray(org_Dataset_x4.test_name_x4))
    #     idx_x4=np.argsort(np.asarray(x4_Dataset.eval_names_x4))
    #     org_idx_x8=np.argsort(np.asarray(org_Dataset_x8.test_name_x8))
    #     idx_x8=np.argsort(np.asarray(x8_Dataset.eval_names_x8))
    #     org_idx_x16=np.argsort(np.asarray(org_Dataset_x16.test_name_x16))
    #     idx_x16=np.argsort(np.asarray(x16_Dataset.eval_names_x16))
        
        
    #     print("the ratio org between interpolation")
    #     print((org_conf_mat_x2[org_idx_x2]==_conf_mat_x2[idx_x2]).sum()/len(org_conf_mat_x2))
    #     print((org_conf_mat_x4[org_idx_x4]==_conf_mat_x4[idx_x4]).sum()/len(org_conf_mat_x4))
    #     print((org_conf_mat_x8[org_idx_x8]==_conf_mat_x8[idx_x8]).sum()/len(org_conf_mat_x8))
    #     print((org_conf_mat_x16[org_idx_x16]==_conf_mat_x16[idx_x16]).sum()/len(org_conf_mat_x16))
        
    #     print("check the sequence of label between org and interpolation")
    #     print(org_all_label_x2)
    #     print(_all_label_x2)
        
    #     print(org_Dataset_x2.test_name_x2)
    #     print(x2_Dataset.eval_names_x2)
    #     print(np.sort(np.asarray(org_Dataset_x2.test_name_x2)))
    #     print(np.sort(np.asarray(x2_Dataset.eval_names_x2)))
        
        
    #     exit()
        
    #     print((org_all_label_x2==_all_label_x2).sum()/len(org_all_label_x2))
    #     print((org_all_label_x4==_all_label_x4).sum()/len(org_all_label_x4))
    #     print((org_all_label_x8==_all_label_x8).sum()/len(org_all_label_x8))
    #     print((org_all_label_x16==_all_label_x16).sum()/len(org_all_label_x16))
        
        
    #     exit()
    #     print("")
    
    # print("")
    
    

    # esb_test_demo(org_loader, models, 'org')
    # esb_test_demo(org_loader_x2, models, 'org_x2')
    # esb_test_demo(x2_loader, models, 'x2')
    # esb_test_demo(org_loader_x4, models, 'org_x4')
    # esb_test_demo(x4_loader, models, 'x4')
    # esb_test_demo(org_loader_x8, models, 'org_x8')
    # esb_test_demo(x8_loader, models, 'x8')
    # esb_test_demo(org_loader_x16, models, 'org_x16')
    # esb_test_demo(x16_loader, models, 'x16')







    # cortical thickness demo data 있는 경우만

    org_Dataset = Thickness_Dataset_interpolation_demo(mode='org')
    org_Dataset_x2 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x2')
    org_Dataset_x4 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x4')
    org_Dataset_x8 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x8')
    org_Dataset_x16 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x16')


    x2_Dataset = Thickness_Dataset_interpolation_demo(mode='x2')
    x4_Dataset = Thickness_Dataset_interpolation_demo(mode='x4')
    x8_Dataset = Thickness_Dataset_interpolation_demo(mode='x8')
    x16_Dataset = Thickness_Dataset_interpolation_demo(mode='x16')

    #define dataloader
    org_loader = torch.utils.data.DataLoader(org_Dataset, batch_size=32, shuffle=False, num_workers=8)
    org_loader_x2 = torch.utils.data.DataLoader(org_Dataset_x2, batch_size=32, shuffle=False, num_workers=8)
    org_loader_x4 = torch.utils.data.DataLoader(org_Dataset_x4, batch_size=32, shuffle=False, num_workers=8)
    org_loader_x8 = torch.utils.data.DataLoader(org_Dataset_x8, batch_size=32, shuffle=False, num_workers=8)
    org_loader_x16 = torch.utils.data.DataLoader(org_Dataset_x16, batch_size=32, shuffle=False, num_workers=8)

    x2_loader = torch.utils.data.DataLoader(x2_Dataset, batch_size=32, shuffle=False, num_workers=8)
    x4_loader = torch.utils.data.DataLoader(x4_Dataset, batch_size=32, shuffle=False, num_workers=8)
    x8_loader = torch.utils.data.DataLoader(x8_Dataset, batch_size=32, shuffle=False, num_workers=8)
    x16_loader = torch.utils.data.DataLoader(x16_Dataset, batch_size=32, shuffle=False, num_workers=8)
    
    #define model 
    model1 = Linear_Model()
    model2 = Linear_Model()
    model3 = Linear_Model()
    model4 = Linear_Model()
    model5 = Linear_Model()


    #load model saved model parameter
    model1.load_state_dict(torch.load('./save_demo/epoch_52_acc_0.82.pth'))
    model2.load_state_dict(torch.load('./save_demo/epoch_93_acc_0.82.pth'))
    model3.load_state_dict(torch.load('./save_demo/epoch_107_acc_0.83.pth'))
    model4.load_state_dict(torch.load('./save_demo/epoch_125_acc_0.84.pth'))
    model5.load_state_dict(torch.load('./save_demo/epoch_225_acc_0.84.pth'))
    
    models= [model1,model2,model3,model4,model5]
    #test performance org , x2 , x4, x8,  x16

    for i, model in enumerate(models):
        print(f"model {i}")    
        org_conf_mat     ,  _= test_demo(org_loader, model, 'org')
        org_conf_mat_x2  , org_all_label_x2 = test_demo(org_loader_x2, model, 'org_x2')
        _conf_mat_x2     , _all_label_x2    = test_demo(x2_loader, model, 'x2')
        org_conf_mat_x4  , org_all_label_x4 = test_demo(org_loader_x4, model, 'org_x4')
        _conf_mat_x4     , _all_label_x4    = test_demo(x4_loader, model, 'x4')
        org_conf_mat_x8  , org_all_label_x8 = test_demo(org_loader_x8, model, 'org_x8')
        _conf_mat_x8     , _all_label_x8    = test_demo(x8_loader, model, 'x8')
        org_conf_mat_x16 , org_all_label_x16 = test_demo(org_loader_x16, model, 'org_x16')
        _conf_mat_x16    , _all_label_x16    = test_demo(x16_loader, model, 'x16')
        
        
        
        org_idx_x2=np.argsort(np.asarray(org_Dataset_x2.test_name_x2))
        idx_x2=np.argsort(np.asarray(x2_Dataset.eval_names_x2))
        org_idx_x4=np.argsort(np.asarray(org_Dataset_x4.test_name_x4))
        idx_x4=np.argsort(np.asarray(x4_Dataset.eval_names_x4))
        org_idx_x8=np.argsort(np.asarray(org_Dataset_x8.test_name_x8))
        idx_x8=np.argsort(np.asarray(x8_Dataset.eval_names_x8))
        org_idx_x16=np.argsort(np.asarray(org_Dataset_x16.test_name_x16))
        idx_x16=np.argsort(np.asarray(x16_Dataset.eval_names_x16))
        
        
        print("the ratio org between interpolation")
        print((org_conf_mat_x2[org_idx_x2]==_conf_mat_x2[idx_x2]).sum()/len(org_conf_mat_x2))
        print((org_conf_mat_x4[org_idx_x4]==_conf_mat_x4[idx_x4]).sum()/len(org_conf_mat_x4))
        print((org_conf_mat_x8[org_idx_x8]==_conf_mat_x8[idx_x8]).sum()/len(org_conf_mat_x8))
        print((org_conf_mat_x16[org_idx_x16]==_conf_mat_x16[idx_x16]).sum()/len(org_conf_mat_x16))
        
        print("check the sequence of label between org and interpolation")
        print(org_all_label_x2)
        print(_all_label_x2)
        
        print(org_Dataset_x2.test_name_x2)
        print(x2_Dataset.eval_names_x2)
        print(np.sort(np.asarray(org_Dataset_x2.test_name_x2)))
        print(np.sort(np.asarray(x2_Dataset.eval_names_x2)))
        
        
        exit()
        
        print((org_all_label_x2==_all_label_x2).sum()/len(org_all_label_x2))
        print((org_all_label_x4==_all_label_x4).sum()/len(org_all_label_x4))
        print((org_all_label_x8==_all_label_x8).sum()/len(org_all_label_x8))
        print((org_all_label_x16==_all_label_x16).sum()/len(org_all_label_x16))
        
        
        exit()
        print("")
    
    print("")
    
    

    esb_test_demo(org_loader, models, 'org')
    esb_test_demo(org_loader_x2, models, 'org_x2')
    esb_test_demo(x2_loader, models, 'x2')
    esb_test_demo(org_loader_x4, models, 'org_x4')
    esb_test_demo(x4_loader, models, 'x4')
    esb_test_demo(org_loader_x8, models, 'org_x8')
    esb_test_demo(x8_loader, models, 'x8')
    esb_test_demo(org_loader_x16, models, 'org_x16')
    esb_test_demo(x16_loader, models, 'x16')

    




    # exit()

import os
import sys

import argparse
from turtle import fd

import numpy as np
import torch
import torch.nn as nn
import torchvision


import pandas as pd


from sklearn import svm
import collections

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics




class Thickness_Dataset(torch.utils.data.Dataset): 
  def __init__(self, mode='train' ,  sub_mode = False):
      self.dataset = pd.read_excel('./220407_thickness_final.xlsx')
      self.demo_dataset = pd.read_excel('./CHC_220420.xlsx')
      self.mode = mode
      self.sub_mode = sub_mode
      self.all_data = []
      self.all_label = []

      self.train_data_for_demo =[]
      self.train_demo_for_demo =[]
      self.train_label_for_demo =[]
      self.train_name_for_demo =[]
      
      self.test_data = []
      self.test_demo = []
      self.test_label = []


      self.test_data_for_demo =[]
      self.test_demo_for_demo =[]
      self.test_label_for_demo =[]
      self.test_name_for_demo =[]
      

      
      self.train_names = os.listdir('/mnt/ICT_DATASET_PNG')
      self.test_names = os.listdir('/mnt/ICT_DATASET_EVAL')
      
      self._train=[]
      self._test=[]

      
      # 모든 데이터로 부터 cortical thinkness 추출
      for train_name in self.train_names:
        cur_name= train_name.split('_')      
        cur_name= "_".join(cur_name[:-1])
        for name, num ,Cingulate, Frontal, Parietal, Temporal, Occipital, label \
          in zip(self.dataset['nuc_t1_inServer'],self.dataset['num'] ,self.dataset['Cingulate'], self.dataset['Frontal'], self.dataset['Parietal'], self.dataset['Temporal'], self.dataset['Occipital'] , self.dataset['Group']  ):
            name=name.split('.')[0]
            if cur_name == name:   
              if np.isnan(Cingulate or Frontal or Parietal or Temporal or Occipital ):
                    break
              else: 
                self.all_data.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                self._train.append(name)
                if label == 'ADD':
                      self.all_label.append([1])
                      break
                elif label == 'NC':
                      self.all_label.append([0])
                      break
                else:
                      print("label error")
                      raise RuntimeError

      # demo 정보 있는 데이터만 추출
      for train_name in self.train_names:
            if train_name[0]=='A':
                  cur_name='_'.join(train_name.split('_')[:3])  
            else:
                  cur_name='_'.join(train_name.split('_')[:2])
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
                              
                              self.train_data_for_demo.append(np.array([Cingulate, Frontal, Parietal, Temporal, Occipital]))
                            #   self.train_demo_for_demo.append(np.array([sex, age_pet, edu, pet, apoe_e4]))
                            #   self.train_demo_for_demo.append(np.array([sex, age_pet, edu]))
                              self.train_demo_for_demo.append(np.array([sex, age_pet, edu,apoe_e4]))
                            #   self.train_demo_for_demo.append(np.array([apoe_e4]))
                              self.train_name_for_demo.append(name)
                        if label == 'ADD':
                              self.train_label_for_demo.append([1])
                              break
                        elif label == 'NC':
                              self.train_label_for_demo.append([0])
                              break
                        else:
                              print("label error")
                              raise RuntimeError





      self.train_data = np.asarray(self.all_data)


      _mean=np.mean(self.train_data , axis = 0)
      _std = np.std(self.train_data , axis = 0)

      self.train_data = (self.train_data-_mean) / _std
      
      self.train_label = self.all_label
      
      self.val_data = self.all_data[1300:]
      self.val_label = self.all_label[1300:]
      






      self.train_data_for_demo =  np.asarray(self.train_data_for_demo)
      
      
      _mean_demo=np.mean(self.train_data_for_demo , axis = 0)
      _std_demo = np.std(self.train_data_for_demo , axis = 0)



      self.train_data_for_demo = (self.train_data_for_demo-_mean_demo) / _std_demo
      self.train_demo_for_demo =  np.asarray(self.train_demo_for_demo)

      _age_pet_mean_demo = np.mean(self.train_demo_for_demo, axis=0)[1]
      _edu_mean_demo = np.mean(self.train_demo_for_demo, axis=0)[2]

      _age_pet_std_demo = np.std(self.train_demo_for_demo, axis=0)[1]
      _edu_std_demo = np.std(self.train_demo_for_demo, axis=0)[2]




      self.train_demo_for_demo[:,1] = (self.train_demo_for_demo[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
      self.train_demo_for_demo[:,2] = (self.train_demo_for_demo[:,2]-_edu_mean_demo)/_edu_std_demo


      
                    
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
      
      self.test_data = np.asarray(self.test_data)
      self.test_data = (self.test_data-_mean) / _std


      # demo 정보 있는 데이터만 추출
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



      self.test_data_for_demo =  np.asarray(self.test_data_for_demo)
      self.test_data_for_demo = (self.test_data_for_demo-_mean_demo) / _std_demo

      self.test_demo_for_demo =  np.asarray(self.test_demo_for_demo)

      self.test_demo_for_demo[:,1] = (self.test_demo_for_demo[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
      self.test_demo_for_demo[:,2] = (self.test_demo_for_demo[:,2]-_edu_mean_demo)/_edu_std_demo

      
      

  def __len__(self):
      if self.mode =='train':
            if self.sub_mode=='demo':
                  return len(self.train_data_for_demo)
            else:
                  return len(self.train_data)
      elif self.mode == 'val':
            if self.sub_mode=='demo':
                  return len(self.train_data_for_demo)
            else:
                  return len(self.val_data)
      elif self.mode == 'test':
            if self.sub_mode=='demo':
                  return len(self.test_data_for_demo)
            else:
                  return len(self.test_data)
      else:
        print("mode error")
        raise RuntimeError
                  

  def __getitem__(self, idx):
      if self.mode == 'train':
            if self.sub_mode == 'demo':
                  x = torch.FloatTensor(self.train_data_for_demo[idx])
                  y = torch.FloatTensor(self.train_label_for_demo[idx])
                  d = torch.FloatTensor(self.train_demo_for_demo[idx])
            else:
                  x = torch.FloatTensor(self.train_data[idx])
                  y = torch.FloatTensor(self.train_label[idx])
      elif self.mode == 'val':
            if self.sub_mode == 'demo':
                  x = torch.FloatTensor(self.val_data[idx])
                  y = torch.FloatTensor(self.val_label[idx])
            else:
                  x = torch.FloatTensor(self.val_data[idx])
                  y = torch.FloatTensor(self.val_label[idx])
      elif self.mode == 'test':
            if self.sub_mode == 'demo':
                  x = torch.FloatTensor(self.test_data_for_demo[idx])
                  y = torch.FloatTensor(self.test_label_for_demo[idx])
                  d = torch.FloatTensor(self.test_demo_for_demo[idx])
            else:
                  x = torch.FloatTensor(self.test_data[idx])
                  y = torch.FloatTensor(self.test_label[idx])

      else:
            print("mode error")
            raise RuntimeError
          

      if self.sub_mode:
            return x,y,d
      else:
            return x,y  


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

        for _name , _data , _label, _demo in zip(self.test_name_for_demo , self.test_data_for_demo , self.test_label_for_demo, self.test_demo_for_demo):

            # _name_temp=_name.split("_")
            # _name_temp.remove('t1')
            # _name = '_'.join(_name_temp)

            _data , _demo = np.asarray(_data) , np.asarray(_demo)

            if _name in temp_eval_names_x2:
                self.test_data_x2.append(_data)
                self.test_label_x2.append(_label)
                self.test_demo_x2.append(_demo)
            if _name in temp_eval_names_x4:
                self.test_data_x4.append(_data)
                self.test_label_x4.append(_label)
                self.test_demo_x4.append(_demo)
            if _name in temp_eval_names_x8:
                self.test_data_x8.append(_data)
                self.test_label_x8.append(_label)
                self.test_demo_x8.append(_demo)
            if _name in temp_eval_names_x16:
                self.test_data_x16.append(_data)
                self.test_label_x16.append(_label)
                self.test_demo_x16.append(_demo)


        
        # 위에서 normalizae 했음


        # self.test_data_x2 = np.asarray(self.test_data_x2)
        # self.test_data_x2 = (self.test_data_x2-_mean_demo) / _std_demo

        # self.test_data_x4 = np.asarray(self.test_data_x4)
        # self.test_data_x4 = (self.test_data_x4-_mean_demo) / _std_demo

        # self.test_data_x8 = np.asarray(self.test_data_x8)
        # self.test_data_x8 = (self.test_data_x8-_mean_demo) / _std_demo

        # self.test_data_x16 = np.asarray(self.test_data_x16)
        # self.test_data_x16 = (self.test_data_x16-_mean_demo) / _std_demo

        # self.test_demo_x2 = np.asarray(self.test_demo_x2)
        # self.test_demo_x2[:,1] = (self.test_demo_x2[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        # self.test_demo_x2[:,2] = (self.test_demo_x2[:,2]-_edu_mean_demo)/_edu_std_demo

        # self.test_demo_x4 = np.asarray(self.test_demo_x4)
        # self.test_demo_x4[:,1] = (self.test_demo_x4[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        # self.test_demo_x4[:,2] = (self.test_demo_x4[:,2]-_edu_mean_demo)/_edu_std_demo

        # self.test_demo_x8 = np.asarray(self.test_demo_x8)
        # self.test_demo_x8[:,1] = (self.test_demo_x8[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        # self.test_demo_x8[:,2] = (self.test_demo_x8[:,2]-_edu_mean_demo)/_edu_std_demo

        # self.test_demo_x16 = np.asarray(self.test_demo_x16)
        # self.test_demo_x16[:,1] = (self.test_demo_x16[:,1]-_age_pet_mean_demo)/_age_pet_std_demo
        # self.test_demo_x16[:,2] = (self.test_demo_x16[:,2]-_edu_mean_demo)/_edu_std_demo


        # print(self.test_data_x2)
        # print(self.test_demo_x2)

        









    

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





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()



    # SVM cortical thickness만 활용한 results
    
    #load Dataset

    # print("setting train datasets")
    
    # train_Dataset = Thickness_Dataset(mode='train', sub_mode=False)
    
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
    

    

    # X = train_Dataset.train_data
    # Y = np.squeeze(np.asarray(train_Dataset.train_label))



    # clf = svm.SVC(class_weight='balanced', gamma = 'auto')
    # clf.fit(X,Y)



    # labels = np.squeeze(np.asarray(org_Dataset.test_label))
    # org_x2_labels = np.squeeze(np.asarray(org_Dataset_x2.test_label_x2))  
    # _x2_labels = np.squeeze(np.asarray(x2_Dataset.eval_label_x2))
    # org_x4_labels = np.squeeze(np.asarray(org_Dataset_x4.test_label_x4))   
    # _x4_labels = np.squeeze(np.asarray(x4_Dataset.eval_label_x4))
    # org_x8_labels = np.squeeze(np.asarray(org_Dataset_x8.test_label_x8))   
    # _x8_labels = np.squeeze(np.asarray(x8_Dataset.eval_label_x8))
    # org_x16_labels = np.squeeze(np.asarray(org_Dataset_x16.test_label_x16))   
    # _x16_labels = np.squeeze(np.asarray(x16_Dataset.eval_label_x16)) 

    # org_preds = np.asarray(clf.predict(org_Dataset.test_data))
    # org_x2_preds = np.asarray(clf.predict(org_Dataset_x2.test_data_x2))
    # _x2_preds = np.asarray(clf.predict(x2_Dataset.eval_data_x2))
    # org_x4_preds = np.asarray(clf.predict(org_Dataset_x4.test_data_x4))
    # _x4_preds = np.asarray(clf.predict(x4_Dataset.eval_data_x4))
    # org_x8_preds = np.asarray(clf.predict(org_Dataset_x8.test_data_x8))
    # _x8_preds = np.asarray(clf.predict(x8_Dataset.eval_data_x8))
    # org_x16_preds = np.asarray(clf.predict(org_Dataset_x16.test_data_x16))
    # _x16_preds = np.asarray(clf.predict(x16_Dataset.eval_data_x16))


    # # print(preds)
    # # print(labels)
    # print((org_preds==labels).sum()/len(org_preds))

    # print((org_x2_preds==org_x2_labels).sum()/len(org_x2_preds))
    # print((_x2_preds==_x2_labels).sum()/len(_x2_preds))

    # print((org_x4_preds==org_x4_labels).sum()/len(org_x4_preds))
    # print((_x4_preds==_x4_labels).sum()/len(_x4_preds))

    # print((org_x8_preds==org_x8_labels).sum()/len(org_x8_preds))
    # print((_x8_preds==_x8_labels).sum()/len(_x8_preds))

    # print((org_x16_preds==org_x16_labels).sum()/len(org_x16_preds))
    # print((_x16_preds==_x16_labels).sum()/len(_x16_preds))



    # print("len")
    # print(len(org_preds))
    # print("x2")
    # print(len(_x2_preds))
    # print((_x2_labels==0).sum())
    # print((_x2_labels==1).sum())
    # print("x4")
    # print(len(_x4_preds))
    # print((_x4_labels==0).sum())
    # print((_x4_labels==1).sum())
    # print("x8")
    # print(len(_x8_preds))
    # print((_x8_labels==0).sum())
    # print((_x8_labels==1).sum())
    # print("x16")
    # print(len(_x16_preds))
    # print((_x16_labels==0).sum())
    # print((_x16_labels==1).sum())






    # SVM cortical thickness  활용한 results (demo 달린 case만)
    
    #load Dataset


    print("setting train datasets")

    train_Dataset = Thickness_Dataset(mode='train', sub_mode='demo')

    print("setting eval datasets")

    org_Dataset = Thickness_Dataset_interpolation_demo(mode='org')
    org_Dataset_x2 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x2')
    org_Dataset_x4 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x4')
    org_Dataset_x8 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x8')
    org_Dataset_x16 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x16')


    x2_Dataset = Thickness_Dataset_interpolation_demo(mode='x2')
    x4_Dataset = Thickness_Dataset_interpolation_demo(mode='x4')
    x8_Dataset = Thickness_Dataset_interpolation_demo(mode='x8')
    x16_Dataset = Thickness_Dataset_interpolation_demo(mode='x16')

    
    
    

    X = train_Dataset.train_data_for_demo
    Y = np.squeeze(np.asarray(train_Dataset.train_label_for_demo))

    print("train SVM")

    clf = svm.SVC(class_weight='balanced', gamma = 'auto', probability=True)
    clf.fit(X,Y)



    labels = np.squeeze(np.asarray(org_Dataset.test_label_for_demo))
    org_x2_labels = np.squeeze(np.asarray(org_Dataset_x2.test_label_x2))  
    _x2_labels = np.squeeze(np.asarray(x2_Dataset.eval_label_x2))
    org_x4_labels = np.squeeze(np.asarray(org_Dataset_x4.test_label_x4))   
    _x4_labels = np.squeeze(np.asarray(x4_Dataset.eval_label_x4))
    org_x8_labels = np.squeeze(np.asarray(org_Dataset_x8.test_label_x8))   
    _x8_labels = np.squeeze(np.asarray(x8_Dataset.eval_label_x8))
    org_x16_labels = np.squeeze(np.asarray(org_Dataset_x16.test_label_x16))   
    _x16_labels = np.squeeze(np.asarray(x16_Dataset.eval_label_x16)) 

    org_preds = np.asarray(clf.predict(org_Dataset.test_data_for_demo))
    org_x2_preds = np.asarray(clf.predict(org_Dataset_x2.test_data_x2))
    _x2_preds = np.asarray(clf.predict(x2_Dataset.eval_data_x2))
    org_x4_preds = np.asarray(clf.predict(org_Dataset_x4.test_data_x4))
    _x4_preds = np.asarray(clf.predict(x4_Dataset.eval_data_x4))
    org_x8_preds = np.asarray(clf.predict(org_Dataset_x8.test_data_x8))
    _x8_preds = np.asarray(clf.predict(x8_Dataset.eval_data_x8))
    org_x16_preds = np.asarray(clf.predict(org_Dataset_x16.test_data_x16))
    _x16_preds = np.asarray(clf.predict(x16_Dataset.eval_data_x16))
    
    
    org_scores     = np.asarray(clf.predict_proba((org_Dataset.test_data_for_demo)))[:,1]
    org_x2_scores  = np.asarray(clf.predict_proba((org_Dataset.test_data_x2      )))[:,1]
    _x2_scores     = np.asarray(clf.predict_proba((org_Dataset.eval_data_x2      )))[:,1]
    org_x4_scores  = np.asarray(clf.predict_proba((org_Dataset.test_data_x4      )))[:,1]
    _x4_scores     = np.asarray(clf.predict_proba((org_Dataset.eval_data_x4      )))[:,1]
    org_x8_scores  = np.asarray(clf.predict_proba((org_Dataset.test_data_x8      )))[:,1]
    _x8_scores     = np.asarray(clf.predict_proba((org_Dataset.eval_data_x8      )))[:,1]
    org_x16_scores = np.asarray(clf.predict_proba((org_Dataset.test_data_x16     )))[:,1]
    _x16_scores    = np.asarray(clf.predict_proba((org_Dataset.eval_data_x16     )))[:,1]



    print("the results of acc")

    # print(preds)
    # print(labels)
    print((org_preds==labels).sum()/len(org_preds))

    print((org_x2_preds==org_x2_labels).sum()/len(org_x2_preds))
    print((_x2_preds==_x2_labels).sum()/len(_x2_preds))

    print((org_x4_preds==org_x4_labels).sum()/len(org_x4_preds))
    print((_x4_preds==_x4_labels).sum()/len(_x4_preds))

    print((org_x8_preds==org_x8_labels).sum()/len(org_x8_preds))
    print((_x8_preds==_x8_labels).sum()/len(_x8_preds))

    print((org_x16_preds==org_x16_labels).sum()/len(org_x16_preds))
    print((_x16_preds==_x16_labels).sum()/len(_x16_preds))


    
    
    print("the results of auc")

    org_auc    = roc_auc_score(labels , org_scores)
    org_x2_auc = roc_auc_score(org_x2_labels , org_x2_scores)
    _x2_auc    = roc_auc_score(_x2_labels , _x2_scores)
    org_x4_auc = roc_auc_score(org_x4_labels , org_x4_scores)
    _x4_auc    = roc_auc_score(_x4_labels , _x4_scores)
    org_x8_auc = roc_auc_score(org_x8_labels , org_x8_scores)
    _x8_auc    = roc_auc_score(_x8_labels , _x8_scores)
    org_x16_auc= roc_auc_score(org_x16_labels , org_x16_scores)
    _x16_auc   = roc_auc_score(_x16_labels , _x16_scores)
    
    print(org_auc)
    print(org_x2_auc)
    print(_x2_auc)
    print(org_x4_auc)
    print(_x4_auc)
    print(org_x8_auc)
    print(_x8_auc)
    print(org_x16_auc)
    print(_x16_auc)
    
    print("")
    

    print("len")
    print(len(org_preds))
    print("x2")
    print(len(_x2_preds))
    print((_x2_labels==0).sum())
    print((_x2_labels==1).sum())
    print("x4")
    print(len(_x4_preds))
    print((_x4_labels==0).sum())
    print((_x4_labels==1).sum())
    print("x8")
    print(len(_x8_preds))
    print((_x8_labels==0).sum())
    print((_x8_labels==1).sum())
    print("x16")
    print(len(_x16_preds))
    print((_x16_labels==0).sum())
    print((_x16_labels==1).sum())



    # SVM cortical thickness & demo 활용한 results 
    
    #load Dataset


    # print("setting train datasets")

    # train_Dataset = Thickness_Dataset(mode='train', sub_mode='demo')

    # print("setting eval datasets")

    # org_Dataset = Thickness_Dataset_interpolation_demo(mode='org')
    # org_Dataset_x2 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x2')
    # org_Dataset_x4 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x4')
    # org_Dataset_x8 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x8')
    # org_Dataset_x16 = Thickness_Dataset_interpolation_demo(mode='org', sub_mode='x16')


    # x2_Dataset = Thickness_Dataset_interpolation_demo(mode='x2')
    # x4_Dataset = Thickness_Dataset_interpolation_demo(mode='x4')
    # x8_Dataset = Thickness_Dataset_interpolation_demo(mode='x8')
    # x16_Dataset = Thickness_Dataset_interpolation_demo(mode='x16')

    
    
    

    # X = train_Dataset.train_data_for_demo
    # D = train_Dataset.train_demo_for_demo

    # C = np.concatenate((X,D), axis =1)
    # print("concat result")
    # print(C.shape)

    # Y = np.squeeze(np.asarray(train_Dataset.train_label_for_demo))

    # print("train SVM")

    # clf = svm.SVC(class_weight='balanced', gamma = 'auto', probability = True)
    # clf.fit(C,Y)



    # labels = np.squeeze(np.asarray(org_Dataset.test_label_for_demo))
    # org_x2_labels = np.squeeze(np.asarray(org_Dataset_x2.test_label_x2))  
    # _x2_labels = np.squeeze(np.asarray(x2_Dataset.eval_label_x2))
    # org_x4_labels = np.squeeze(np.asarray(org_Dataset_x4.test_label_x4))   
    # _x4_labels = np.squeeze(np.asarray(x4_Dataset.eval_label_x4))
    # org_x8_labels = np.squeeze(np.asarray(org_Dataset_x8.test_label_x8))   
    # _x8_labels = np.squeeze(np.asarray(x8_Dataset.eval_label_x8))
    # org_x16_labels = np.squeeze(np.asarray(org_Dataset_x16.test_label_x16))   
    # _x16_labels = np.squeeze(np.asarray(x16_Dataset.eval_label_x16)) 
    
    # print(clf.predict_proba(np.concatenate((org_Dataset.test_data_for_demo,org_Dataset.test_demo_for_demo), axis=1)))
    # print(clf.predict_proba(np.concatenate((org_Dataset.test_data_for_demo,org_Dataset.test_demo_for_demo), axis=1)).shape)
    
    
    
    
    

    # org_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.test_data_for_demo,org_Dataset.test_demo_for_demo), axis=1)))
    # org_x2_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.test_data_x2,org_Dataset.test_demo_x2), axis=1)))
    # _x2_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.eval_data_x2,org_Dataset.eval_demo_x2), axis=1)))
    # org_x4_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.test_data_x4,org_Dataset.test_demo_x4), axis=1)))
    # _x4_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.eval_data_x4,org_Dataset.eval_demo_x4), axis=1)))
    # org_x8_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.test_data_x8,org_Dataset.test_demo_x8), axis=1)))
    # _x8_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.eval_data_x8,org_Dataset.eval_demo_x8), axis=1)))
    # org_x16_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.test_data_x16,org_Dataset.test_demo_x16), axis=1)))
    # _x16_preds = np.asarray(clf.predict(np.concatenate((org_Dataset.eval_data_x16,org_Dataset.eval_demo_x16), axis=1)))

    # org_scores     = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.test_data_for_demo,org_Dataset.test_demo_for_demo), axis=1)))[:,1]
    # org_x2_scores  = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.test_data_x2,org_Dataset.test_demo_x2), axis=1)))[:,1]
    # _x2_scores     = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.eval_data_x2,org_Dataset.eval_demo_x2), axis=1)))[:,1]
    # org_x4_scores  = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.test_data_x4,org_Dataset.test_demo_x4), axis=1)))[:,1]
    # _x4_scores     = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.eval_data_x4,org_Dataset.eval_demo_x4), axis=1)))[:,1]
    # org_x8_scores  = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.test_data_x8,org_Dataset.test_demo_x8), axis=1)))[:,1]
    # _x8_scores     = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.eval_data_x8,org_Dataset.eval_demo_x8), axis=1)))[:,1]
    # org_x16_scores = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.test_data_x16,org_Dataset.test_demo_x16), axis=1)))[:,1]
    # _x16_scores    = np.asarray(clf.predict_proba(np.concatenate((org_Dataset.eval_data_x16,org_Dataset.eval_demo_x16), axis=1)))[:,1]


    # # print(preds)
    # # print(labels)
    
    
    # print("the results acc")
    
    # print((org_preds==labels).sum()/len(org_preds))

    # print((org_x2_preds==org_x2_labels).sum()/len(org_x2_preds))
    # print((_x2_preds==_x2_labels).sum()/len(_x2_preds))

    # print((org_x4_preds==org_x4_labels).sum()/len(org_x4_preds))
    # print((_x4_preds==_x4_labels).sum()/len(_x4_preds))

    # print((org_x8_preds==org_x8_labels).sum()/len(org_x8_preds))
    # print((_x8_preds==_x8_labels).sum()/len(_x8_preds))

    # print((org_x16_preds==org_x16_labels).sum()/len(org_x16_preds))
    # print((_x16_preds==_x16_labels).sum()/len(_x16_preds))
    
    # print("")
    
    # print("the results of auc")

    # org_auc    = roc_auc_score(labels , org_scores)
    # org_x2_auc = roc_auc_score(org_x2_labels , org_x2_scores)
    # _x2_auc    = roc_auc_score(_x2_labels , _x2_scores)
    # org_x4_auc = roc_auc_score(org_x4_labels , org_x4_scores)
    # _x4_auc    = roc_auc_score(_x4_labels , _x4_scores)
    # org_x8_auc = roc_auc_score(org_x8_labels , org_x8_scores)
    # _x8_auc    = roc_auc_score(_x8_labels , _x8_scores)
    # org_x16_auc= roc_auc_score(org_x16_labels , org_x16_scores)
    # _x16_auc   = roc_auc_score(_x16_labels , _x16_scores)
    
    # print(org_auc)
    # print(org_x2_auc)
    # print(_x2_auc)
    # print(org_x4_auc)
    # print(_x4_auc)
    # print(org_x8_auc)
    # print(_x8_auc)
    # print(org_x16_auc)
    # print(_x16_auc)
    
    # print("")

    # print("len")
    # print(len(org_preds))
    # print("x2")
    # print(len(_x2_preds))
    # print((_x2_labels==0).sum())
    # print((_x2_labels==1).sum())
    # print("x4")
    # print(len(_x4_preds))
    # print((_x4_labels==0).sum())
    # print((_x4_labels==1).sum())
    # print("x8")
    # print(len(_x8_preds))
    # print((_x8_labels==0).sum())
    # print((_x8_labels==1).sum())
    # print("x16")
    # print(len(_x16_preds))
    # print((_x16_labels==0).sum())
    # print((_x16_labels==1).sum())


    

    
    

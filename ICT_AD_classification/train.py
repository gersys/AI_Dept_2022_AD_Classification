
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




# Thickness Custom dataset





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
                              # self.train_demo_for_demo.append(np.array([sex, age_pet, edu, pet, apoe_e4]))
                              # self.train_demo_for_demo.append(np.array([sex, age_pet, edu]))
                              # self.train_demo_for_demo.append(np.array([sex, age_pet, edu, apoe_e4]))
                              self.train_demo_for_demo.append(np.array([apoe_e4]))
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
                              # self.test_demo_for_demo.append(np.array([sex, age_pet, edu, pet, apoe_e4]))
                              # self.test_demo_for_demo.append(np.array([sex, age_pet, edu]))
                              # self.test_demo_for_demo.append(np.array([sex, age_pet, edu, apoe_e4]))
                              self.test_demo_for_demo.append(np.array([apoe_e4]))
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
            x = self.linear1_3(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.linear3(x)
            return x



def train(e,train_loader, model, criterion, optimizer):
      total_pred = []
      total_label = []
      for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda() , labels.cuda()           
            outputs  = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            labels = torch.squeeze(labels.long())

            total_pred.append(preds.cpu().numpy())
            total_label.append(labels.cpu().numpy())
                  

            loss = criterion(outputs, labels)
            
            # print(f"loss {loss:.4f}")
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # for p in model.parameters():
            #       print(p.data)
            # exit()
      
      total_pred = np.asarray(total_pred)
      total_label = np.asarray(total_label)
      
      total_pred = np.concatenate(total_pred, axis = 0)
      total_label = np.concatenate(total_label, axis = 0)

      cur_acc = (total_pred ==total_label).sum()/len(total_pred)
      


      
      print(f"e: {e} train acc: {cur_acc:.2f}")

def train_demo(e,train_loader, model, criterion, optimizer):
      total_pred = []
      total_label = []
      for i, (inputs, labels, demos) in enumerate(train_loader):
            inputs, labels , demos = inputs.cuda() , labels.cuda() , demos.cuda()

            inputs = torch.cat((inputs, demos), dim=1)

            outputs  = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            labels = torch.squeeze(labels.long())

            total_pred.append(preds.cpu().numpy())
            total_label.append(labels.cpu().numpy())
                  

            loss = criterion(outputs, labels)
            
            # print(f"loss {loss:.4f}")
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # for p in model.parameters():
            #       print(p.data)
            # exit()
      
      total_pred = np.asarray(total_pred)
      total_label = np.asarray(total_label)
      
      total_pred = np.concatenate(total_pred, axis = 0)
      total_label = np.concatenate(total_label, axis = 0)

      cur_acc = (total_pred ==total_label).sum()/len(total_pred)
      


      
      print(f"e: {e} train acc: {cur_acc:.2f}")              
            
            
            
            
            
      




def val(e, val_loader, model, criterion, optimizer, best_acc):
      total = 0
      ans = 0
      model=model.cuda()
      model.eval()
      with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
              inputs , labels = inputs.cuda() , labels.cuda()
              outputs=model(inputs)
              
              outputs = torch.round(outputs)
            
              
              ans+=(outputs==labels).sum()
              total+=inputs.shape[0]
              
      
      cur_acc = ans/total 
      
      print(f"e: {e} val acc: {cur_acc:.2f}")        
      
      
      save_PATH = './save/'
      
      os.makedirs(save_PATH, exist_ok=True)
      
      if cur_acc >= best_acc:
            
            torch.save(model.state_dict(), save_PATH+f"epoch_{e}_acc_{cur_acc:.2f}.pth" )
            best_acc = cur_acc
            print("model saved")
      
      return best_acc
              


 
def test(e, test_loader, model, criterion, optimizer, best_acc):
      
      total_pred = []
      total_label = []
      with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                  inputs , labels = inputs.cuda() , labels.cuda()


                  outputs=model(inputs)



                  preds = torch.argmax(outputs, dim=1)

                  labels = torch.squeeze(labels.long())

                  total_pred.append(preds.cpu().numpy())
                  total_label.append(labels.cpu().numpy())
                  
      
      total_pred = np.asarray(total_pred)
      total_label = np.asarray(total_label)
      
      total_pred = np.concatenate(total_pred, axis = 0)
      total_label = np.concatenate(total_label, axis = 0)

      cur_acc = (total_pred ==total_label).sum()/len(total_pred)
      
      print(f"e: {e} test acc: {cur_acc:.2f}")        

      save_PATH = './save/'
      
      os.makedirs(save_PATH, exist_ok=True)
      
      if cur_acc >= best_acc:
            
            torch.save(model.state_dict(), save_PATH+f"epoch_{e}_acc_{cur_acc:.2f}.pth" )
            best_acc = cur_acc
            print("model saved")
      
              
      return best_acc


def test_demo(e, test_loader, model, criterion, optimizer, best_acc):
      
      total_pred = []
      total_label = []
      with torch.no_grad():
            for i, (inputs, labels , demos) in enumerate(test_loader):
                  inputs , labels, demos = inputs.cuda() , labels.cuda() , demos.cuda()
                  inputs = torch.cat((inputs, demos), dim=1)

                  outputs=model(inputs)
                  



                  preds = torch.argmax(outputs, dim=1)

                  labels = torch.squeeze(labels.long())

                  total_pred.append(preds.cpu().numpy())
                  total_label.append(labels.cpu().numpy())
                  
      
      total_pred = np.asarray(total_pred)
      total_label = np.asarray(total_label)
      
      total_pred = np.concatenate(total_pred, axis = 0)
      total_label = np.concatenate(total_label, axis = 0)

      cur_acc = (total_pred ==total_label).sum()/len(total_pred)
      
      print(f"e: {e} test acc: {cur_acc:.2f}")        

      save_PATH = './save_demo/'
      
      os.makedirs(save_PATH, exist_ok=True)
      
      if cur_acc >= best_acc:
            
            torch.save(model.state_dict(), save_PATH+f"epoch_{e}_acc_{cur_acc:.2f}.pth" )
            best_acc = cur_acc
            print("model saved")
      
              
      return best_acc
              

              






if __name__=="__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--epochs", default=1000)
      parser.add_argument("--sub_mode", default=False)

      args = parser.parse_args()

      #load Dataset

      print("setting train dataset")
      train_Dataset = Thickness_Dataset(mode='train', sub_mode=args.sub_mode)
      print("setting val dataset")
      # val_Dataset = Thickness_Dataset(mode='val', sub_mode=args.sub_mode)
      print("setting test dataset")
      test_Dataset = Thickness_Dataset(mode='test', sub_mode=args.sub_mode)

      

      #define dataloader
      train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=64, shuffle=True, num_workers=8)
      # val_loader = torch.utils.data.DataLoader(val_Dataset, batch_size=16, shuffle=False, num_workers=2)
      test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=16, shuffle=False, num_workers=2)

      #define model 
      if args.sub_mode == 'demo':
            model = Linear_Model_demo()
            # model = Linear_Model()
      else:
            model = Linear_Model()

      #define loss function
      # criterion = torch.nn.BCELoss()
      criterion = torch.nn.CrossEntropyLoss()

      #define optimizer
      # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01 , momentum = 0.9 , weight_decay=0.9)
      optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.9)
      LR_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100,1000])
      
      model = model.cuda()
      best_acc = 0
      for e in range(args.epochs):
            model.train()
            if args.sub_mode=='demo':
                  train_demo(e, train_loader, model, criterion, optimizer)
            else:
                  train(e, train_loader, model, criterion, optimizer)
            model.eval()
            # best_acc = val(e, val_loader, model, criterion, optimizer, best_acc)
            if args.sub_mode=='demo':
                  best_acc= test_demo(e, test_loader, model, criterion, optimizer, best_acc)
            else:
                  best_acc= test(e, test_loader, model, criterion, optimizer, best_acc)
            print(f"current Learning rate schedule is {LR_scheduler.get_last_lr()}")
            LR_scheduler.step()
    
    
          
    
    
    
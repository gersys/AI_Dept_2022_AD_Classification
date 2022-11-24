import torch
from torch import nn as nn
from sklearn import svm

class MLP(torch.nn.Module):
    
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 100
        self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.linear3 = torch.nn.Linear(self.hidden_dim, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_dim)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
        
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                torch.nn.init.kaiming_normal_(mod.weight.data)
                if mod.bias is not None:
                    mod.bias.data.zero_()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


class SVM_wrapper():
    def __init__(self):
        self.svm = svm.SVC(class_weight='balanced', gamma='auto', probability=True)
        
    def fit(self, x, y):
        return self.svm.fit(x,y)
        
    def predict(self, x):
        return self.svm.predict(x)
    
    def predict_proba(self, x):
        return self.svm.predict_proba(x)
    
    
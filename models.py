import torch
import torch.nn.functional as F

class DropoutModel(torch.nn.Module):
    def __init__(self):
        super(DropoutModel,self).__init__()
        self.layer1 = torch.nn.Linear(784,1200)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.layer2 = torch.nn.Linear(1200,1200)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.layer3 = torch.nn.Linear(1200,10)
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)

class FashionDNN(torch.nn.Module):
    def __init__(self):
        super(FashionDNN,self).__init__()
        self.fc1 = torch.nn.Linear(in_features=784,out_features=256)
        self.drop = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(in_features=256,out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128,out_features=10)

    def forward(self,input_data):
        out = input_data.view(-1,784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

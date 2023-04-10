import torch.nn as nn
import torch.nn.functional as F



class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride =  1)
        self.bn1 = nn.BatchNorm2d(num_features = 16)
        self.mp1 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride =  1)
        self.bn2 = nn.BatchNorm2d(num_features = 32)
        self.mp2 = nn.MaxPool2d(kernel_size = 2)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1)
        self.mp3 = nn.MaxPool2d(kernel_size = 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64,100)
        self.fc2 = nn.Linear(100,10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mp2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.mp3(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x  
    
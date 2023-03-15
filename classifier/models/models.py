import torch.nn as nn
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchmetrics.functional import accuracy
import os
import sys

base_path = os.path.join(os.path.dirname(__file__),'base')
sys.path.append(base_path)

from base_model import BaseModel


def conv_block(in_channels, out_channels, pool = False):
    """
    Function to create the convolutional block for the models
    
    Args:
        in_channels (_int_): number of input channels of the block
        out_channels (_type_): number of the output channels of the block
        pool (bool, optional): If True adds a MaxPool2d layer in the final of the block Defaults to False.

    Returns:
        _Sequential_: returns a sequential model with the specified configs
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers) 

class ResNet12(BaseModel,LightningModule):
    """
    Class to implement ResNet12 architecture, extends from BaseModel class that is reponsible for implement the basics
    model evaluations methods.
    More details of ResNet12 architecture can be found on:
    https://www.researchgate.net/figure/The-structure-of-ResNet-12_fig1_329954455

    Args:
        BaseModel (Class): Class that is reponsible for implement the basics model evaluations methods
        LightningModule (Class): Class from pytorch-ligthining that have abstractions to handle with metrics and use auto learning rate finder
    """
    def __init__(self,in_channels,num_classes,learning_rate):
        """
        Constructor of the class

        Args:
            in_channels (_int_): Size of the input channels (has to match with the pixels dimensions of the images)
            num_classes (_int_): Number of classes that will be classified, specifies the size of the output layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.lr = learning_rate
        # 3 x 224 x 224
        self.conv1= conv_block(in_channels,16) # 16 x 224 x 224
        self.conv2 = conv_block(16,32,pool=True) # 32 x 112 x 112
        self.res1 = nn.Sequential(conv_block(32,32),conv_block(32,32)) # 32 x 112 x 112
        self.conv3 = conv_block(32,64,pool=True) # 64 x 56 x 56
        self.conv4 = conv_block(64,128,pool=True) # 128 x 28 x 28
        self.res2 =  nn.Sequential(conv_block(128,128),conv_block(128,128)) # 128 x 28 X 28
        self.conv5 = conv_block(128,192,pool=True) # 192 x 14 x 14
        self.conv6 = conv_block(192,256,pool= True) # 256 x 6 x 6
        self.res3 = nn.Sequential(conv_block(256,256),conv_block(256,256))
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 256 x 3 x 3
                                        nn.Flatten(), # 256 * 3* 3
                                        nn.Linear(256*3*3,256*3*3),
                                        nn.Dropout(0.2), # 256
                                        nn.Linear(256*3*3,num_classes)) #19
    
    def forward(self,xb):
        """
        Method to flow the data foward the network architecture

        Args:
            xb (_list_): batch of the data that will be used in the step

        Returns:
            _list_: list with the predictions maded by the model
        """
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),  lr=(self.lr or self.learning_rate))
    
    

class LeNet(BaseModel,LightningModule):
    """
    Class to implement LeNet12 architecture, extends from BaseModel class that is reponsible for implement the basics
    model evaluations methods.
    More details of LeNet12 architecture can be found on:

    Args:
        BaseModel (Class): Class that is reponsible for implement the basics model evaluations methods
        LightningModule (Class): Class from pytorch-ligthining that have abstractions to handle with metrics and use auto learning rate finder
    """
    def __init__(self, in_channels, n_classes,learning_rate):
        super().__init__()
        self.num_classes = n_classes
        self.lr = learning_rate
        # fist conv
        self.conv1 = conv_block(in_channels,16) # 16 x 224 x 224
        self.conv2 = conv_block(16,32, pool=True) # 64 x 112 x  112
        # second conv
        self.conv3 = conv_block(32,64, pool=True) # 128 x 56 x 56
        self.conv4 = conv_block(64,128, pool= True) # 192 x 28 x 28
        self.conv5 = conv_block(128,192, pool=True) # 256 x 14 x 14
        self.conv6 = conv_block(192, 256, pool= True) # 256 x 6 x 6
        # fist and only set of FC
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 256 x 3 x3
        nn.Flatten(),     
        nn.Linear(in_features= 256 * 3 * 3, out_features= 256 *3 *3 ), # 256 x 3 x 3
                    nn.ReLU(),
                    nn.Linear(in_features = 256 * 3 * 3, out_features = n_classes),
                    nn.LogSoftmax(dim=1)
                    )

    
        
    def forward(self,xb):
        """
        foward method to pass the data network throug architeture 

        Returns:
            _list_: the output predictions
        """
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.classifier(out)
        return out
    

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = (self.lr or self.learning_rate))
    
    



class AlexNet(BaseModel,LightningModule):
    """
    Class to implement AlexNet architecture, extends from BaseModel , and LighthingModule.
    More details of AlexNet architecture can be found on:
    https://arxiv.org/abs/1803.01164
    

    Args:
        BaseModel (Class): Class that is reponsible for implement the basics model evaluations methods
        LightningModule (Class): Class from pytorch-ligthining that have abstractions to handle with metrics and use auto learning rate finder
    """
    def __init__(self, in_channels, n_classes,learning_rate):
        super().__init__()
        self.num_classes = n_classes
        self.lr = learning_rate
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
                                    )
        
        self.layer2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)
                                    )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(384,384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, n_classes)
        )
        
    def forward(self, xb):
        out = self.layer1(xb)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = (self.lr or self.learning_rate))
    



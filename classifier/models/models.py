import torch.nn as nn
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchmetrics.functional import accuracy


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

class ResNet12(LightningModule):
    """
    Class to implement ResNet12 architecture, extends from ImageClassificationBase class that is reponsible to implemente the basics
    model evaluations methods.
    More details of ResNet12 architecture can be found on:
    https://www.researchgate.net/figure/The-structure-of-ResNet-12_fig1_329954455

    Args:
        ImageClassificationBase (_Class_): Base class with the methods to evaluation the model
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
    
    def training_step(self, batch):
        """
        Method to get the loss of the training step of the model

        Args:
            batch (_DataLoader object_): batch of images that will be used in training step

        Returns:
            _float_: returns the loss of step
        """
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        self.log("train_loss",loss, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Method to get the validation loss and validation accuracy for the step of the model

        Args:
            batch (_DataLoader object_): batch of images that will be used in training step

        Returns:
            _dict_: returns a dict with the validation loss and validation accuracy of the step 
        """
        images, labels = batch
        
        out = self(images)
        
        loss = F.cross_entropy(out, labels)
        if self.num_classes == 2:
            acc = accuracy(out, labels,task='binary')
        else:
            acc = accuracy(out,labels, task='multiclass',num_classes=self.num_classes)
        
        self.log("val_loss",loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        
        return {'val_loss':loss.detach(), 'val_acc': acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    

class LeNet(LightningModule):
    
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
    
    def training_step(self, batch):
        """
        Method to get the loss of the training step of the model

        Args:
            batch (_DataLoader object_): batch of images that will be used in training step

        Returns:
            _float_: returns the loss of step
        """
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        self.log("train_loss",loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Method to get the validation loss and validation accuracy for the step of the model

        Args:
            batch (_DataLoader object_): batch of images that will be used in training step

        Returns:
            _dict_: returns a dict with the validation loss and validation accuracy of the step 
        """
        images, labels = batch
        
        out = self(images)
        
        loss = F.cross_entropy(out, labels)
        if self.num_classes == 2:
            acc = accuracy(out, labels,task='binary')
        else:
            acc = accuracy(out,labels, task='multiclass',num_classes=self.num_classes)
        
        self.log("val_loss",loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        
        return {'val_loss':loss.detach(), 'val_acc': acc}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
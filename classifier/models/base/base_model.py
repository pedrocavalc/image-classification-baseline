import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torch.nn as nn
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch
class BaseModel():
    """
    Base class for instanced models to handle methods common to classes that implement Network architectures.
    """
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
    


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out
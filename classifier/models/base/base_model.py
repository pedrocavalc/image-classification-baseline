import torch.nn.functional as F
from torchmetrics.functional import accuracy

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
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    """
    Function to calculate the accuracy of the model predictions

    Args:
        outputs (_list_): list of predictions of the model
        labels (_list_): list of the real labels for the images
    Returns:
        tensor: tensor with the value of accuracy
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
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
        return loss
    
    
    def validation_step(self, batch):
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
        
        acc = accuracy(out, labels)
        
        return {'val_loss':loss.detach(), 'val_acc': acc}
    
    
    def validation_epoch_end(self, outputs):
        """
        Method to get the loss and accuracy at the end of a epoch
        Args:
            outputs (_list_): list of dicts with de validation losses and validation accuracys for each step of the epoch

        Returns:
            _dict_:  dict with the validation loss and accuracy for the entire epoch
        """
        batch_losses = [x['val_loss']for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # combine accuracys
        return {'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()}
    
    

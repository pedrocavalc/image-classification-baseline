import os
from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import sys
utils_path = os.path.join(os.path.dirname(__file__),'../utils')
sys.path.append(utils_path)


from image_visualization import show_images , show_batch
from classifier.gpu_management import get_default_device, DeviceDataLoader, to_device
from classifier.models.models import ResNet12
from classifier.fit_cycle import fit_one_cycle


def data_augmentation():
    """
    Function responsible for configuring the data augmentation in the dataset.
    The data augmentations used by default are: 
    Training set: RandomHorizontalflip, Resize(default to 224 x 224), Normalize with predefined statuses
    Validation set: Normalization, with predefined statuses.    
    Returns:
        Compose pytorch objects: returns two transformers for the data, 
        one for the training set and the other for the validation set
    """
    stats = ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)) # stats to normalize image
    train_transforms = tt.Compose([ tt.RandomHorizontalFlip(),tt.ToTensor(), 
                                   tt.Resize(size=(224,224)),tt.Normalize(*stats,inplace=True)]) # augmentation applied
    val_transforms = tt.Compose([tt.ToTensor(),tt.Resize(size=(224,224)),tt.Normalize(*stats)])
    return train_transforms, val_transforms



def train(data_dir, augmentation= True, batch_size = 64, epochs=10, max_lr=0.5):
    """ 
    
    Function to train models in pytorch, the path to the dataset that will be used for training 
    is passed and data augmentation will be applied if so required.
    Args:
        data_dir (str): path to dataset that will be use to train
        augmentation (bool, optional): Whether or not data augmentation will be performed Defaults to True.
        batch_size (int, optional): Size of a mini batch of images that will be used in training
    Returns:
        model: pytorch model trained
    """
    train_path = os.path.join(data_dir,'train') # paths to dataset
    val_path = os.path.join(data_dir, 'valid')
    show_images(train_path) # show images loaded
    
    if augmentation: # if arg augmentation is True applied the augmentation in images
        train_transforms, val_transforms = data_augmentation()
        train_dataset = ImageFolder(train_path,train_transforms)
        val_dataset = ImageFolder(val_path, val_transforms)
    else:
        train_dataset = ImageFolder(train_path)
        val_dataset = ImageFolder(val_path)
    # creating the data loaders   
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size * 2, num_workers=2, pin_memory= True)
    show_batch(train_loader) # show the train_loader batch        
    device_flag = get_default_device() # get the devices availables in machine
    device = torch.device(device_flag)
    train_dl = DeviceDataLoader(train_loader,device) # moving the data to target device
    val_dl = DeviceDataLoader(val_loader, device)
    
    model = to_device(ResNet12(3,500),device)
    print(model)
    history = fit_one_cycle(epochs=epochs, max_lr=max_lr,model=model,train_loader=train_dl, val_loader=val_dl,)
    


if __name__ == "__main__":
    train(data_dir='../data/100-bird-species')
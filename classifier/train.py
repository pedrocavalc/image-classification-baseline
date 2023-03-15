import os
from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import pytorch_lightning as pl
import mlflow.pytorch

import sys
utils_path = os.path.join(os.path.dirname(__file__),'../utils')
sys.path.append(utils_path)



from image_visualization import show_images , show_batch
from classifier.models.models_config import config_models
mlflow.set_tracking_uri("http://127.0.0.1:5000")

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



def train(data_dir,n_classes, augmentation= True, batch_size = 64, epochs=10, max_lr=0.5, n_devices=1, accelerator='cpu'):
    """ 
    
    Function to train models in pytorch, the path to the dataset that will be used for training 
    is passed and data augmentation will be applied if so required. The model and metrics will be automatically logged into mlflow.
    Args:
        data_dir (str): path to dataset that will be use to train
        n_classes (int): Number of classes that have in the dataset.
        augmentation (bool, optional): Whether or not data augmentation will be performed Defaults to True.
        batch_size (int, optional): Size of a mini batch of images that will be used in training
        max_lr (int, optional): Maximum learning rate that will be used in training. Defaults to 0.5.
        n_devices (int, optional): Number of available GPU devices on machine. Defaults to 1
        accelerator (str,optional): Type of the accelerator that are available, set to "cuda" if do you have a GPU with cuda available
        to speed up the training. Defaults to "cpu".
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
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size * 2, num_workers=12, pin_memory= True)
    show_batch(train_loader) # show the train_loader batch
            
    models = config_models(n_classes,max_lr)
    mlflow.pytorch.autolog()
    for key in models:
        model = models[key]
        print(model)
        with mlflow.start_run() as run:
            trainer = pl.Trainer(max_epochs=epochs, devices = n_devices, accelerator = accelerator,auto_lr_find=True)
            trainer.fit(model,train_loader,val_loader)
            mlflow.set_tags({"model":key})



if __name__ == "__main__":
    train(data_dir='../data/100-bird-species')
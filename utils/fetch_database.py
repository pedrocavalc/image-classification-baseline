import os
import tarfile
import matplotlib
import matplotlib.pyplot as plt
import opendatasets as od
import shutil
import logging

class DataSet():
    '''
    Class used to download and manipulate the datasets downloaded from kaagle paths in computer
    '''
    def __init__(self,url,data_dir = "data"):
        '''
        Constructor
        parameters: url ->  link of dataset to be downloaded
        data_dir -> path directory to put the data
        '''
        self.url = url
        self.data_dir = data_dir

    def fech_data(self):
        '''
        Function to download the dataset
        '''
        if os.path.exists(os.path.join(self.data_dir,self.url.split('/')[-1])):
            print(f"Found a existing folder for this dataset in directory '{self.data_dir}")
        else:
            od.download(self.url)
            self.save_dataset()

    def save_dataset(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        path_dataset = self.url.split('/')[-1]

        path_dataset = os.path.join(path_dataset, os.listdir(path_dataset)[0])
        try:
            shutil.move(path_dataset,self.data_dir)
            print(f'Dataset has been downloaded to directory {self.data_dir}')
            os.removedirs(self.url.split('/')[-1])
        except:
            print("Dataset has already been downloaded")
            shutil.rmtree(self.url.split('/')[-1])
            



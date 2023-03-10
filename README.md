# image-classification-baseline

This repository is a system for training image classification models in pytorch using MLOps techniques with easy customization for problems.

## Training on custom dataset

To train the models on your data follow these steps:

1. Extract the dataset into the data folder. 
**Obs: the data must be in the following folder structure:**
  
├── data
│   ├── train
│   │   ├── class_name/*.png
          ...
│   ├── valid
│   │   ├── class_name/*.png
          ...
│   ├── test
│   │   ├── class_name/*.png
        ...
└── .gitignore


## Using MLFlow
To use mlflow go to the project's ROOT directory and use the following command:
```
mlflow ui
```
command in the terminal, the API will be allocated on port 5000 of your machine's localhost. Open your browser and type localhost:5000

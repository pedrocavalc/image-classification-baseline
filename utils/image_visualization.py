import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import make_grid

def recursive_paths_imgs(path_folder):
    """
    Function to find the images paths for visualization with matplotlib
    
    Args:
        path_folder (str): path to folder where the classes folders are
    
    Returns:
        a list with another list with the paths of images from each class in dataset
    """
    class_dirs = glob.glob(path_folder + '/*')
    folders_path = [glob.glob(dirs + '/*')  for dirs in class_dirs]
    images_path = [element for sublist in folders_path for element in sublist]
    return images_path


def show_images(folder_path):
    """ 
    
    function that shows an image of the dataset for exploration
    the function will display every image until the "esc" key is pressed.
    Args:
        folder_path (str): path do folder that has the images from dataset
    """
    path_to_imgs = recursive_paths_imgs(folder_path)
    path_image = random.choice(path_to_imgs)
    img = mpimg.imread(path_image)
    imgplot = plt.imshow(img)
    plt.show()



def show_batch(dl):
    '''
    Function to show the images from a batch in a pytorch loader
    '''
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(16,12))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break

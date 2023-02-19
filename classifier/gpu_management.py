import torch


class DeviceDataLoader():
    """
    Class that will be responsible for move dataloaders into device memory
    
    Constructor params ->:
    data_loader = batch of data that will be moved to device memory
    device = target device
    """
    
    def __init__(self, data_loader, device):
        """
        Constructor

        Args:
            data_loader (DataLoader): data loader pytorch object with the images that will be used
            in training
            device (str): name of devices availables. 
            -> Cuda: if has a GPU available in machine.
            -> CPU: CPU ifs not has a available GPU in machine.
        """
        self.data_loader = data_loader
        self.device = device
        
    def __iter__(self):
        """
        Method that iterate over the data loader to move the data into device

        Yields:
            function: try to use the to_device function to move the data into device
        """
        for batch in self.data_loader:
            yield to_device(batch, self.device)
    
    def __len__(self):
        """
        Number of batches
        """
        return len(self.data_loader)
    
    def __to_device__(self):
        pass


def get_default_device():
    """
    Function to get the devices avaiables on machine

    Returns:
        _str_: "cuda" if has a compatible GPU else "cpu" 
    """
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def to_device(data, device):
    """
    Function to move the data to device

    Args:
        data (list, tuple): iterable of data that will be moved to target device_
        device (str): the device target

    Returns:
        list: return the data allocated on device memory
    """
    if isinstance(data,(list,tuple)):
        return [to_device(image,device ) for image in data]
    return data.to(device, non_blocking=True)

    
    


from typing import Tuple, List
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# root_dir= '../datasets/seg'

def load_batch(root_dir: str, batch_size: int =64, resize: int =150, 
                               crop_size: int =140, degree_rot: int =30)->Tuple[DataLoader,List[str]]:
    
    """
    Return a tuple of DataLoader, batches which will be shuffle dataset with pytorch DataLoader.
    train_loader and test_loader: containing the training tensor transformation images pixels 
    and the labels of the dataset encoded based on the arrangement of the folders containing the images.

    Parameters:
    ----------

    path_dir: str, Required
        Is the path containing the folders where the individual images can found.

    batch_size: int, default set as (64)
        This determine the number of dataset in each batch for the training, and test data

    resize: int, default = 255
        
    """
    
    train_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                        transforms.RandomRotation(degree_rot),
                                        transforms.RandomResizedCrop(crop_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5],
                                                                [0.5, 0.5, 0.5])])

    test_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(root_dir+'/seg_train', transform=train_transforms)
    test_data = datasets.ImageFolder(root_dir+ '/seg_test' , transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    class_list, class_idx = train_data.classes, train_data.class_to_idx

    return train_loader, test_loader, class_list, class_idx



train_loader, test_loader, class_list, class_idx=load_batch('E:\Datasets\seg')


print(class_idx)

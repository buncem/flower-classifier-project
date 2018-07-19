import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

def load_data(data_dir):
    '''
    Load dataloaders with train, validation and test data.
    Transform data into 224x224 torch tensor and normalize to use with pretrained model.
    Perform augmentation on training data including random rotation, random resize and random horizontal flip.
    Get class_to_idx dictionary.
    
    Returns dataloaders and class_to_idx 
    '''
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])]),
                       'validate' : transforms.Compose([transforms.Resize(255),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])])}
    
    image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'validate' : datasets.ImageFolder(valid_dir, transform=data_transforms['validate']),
                      'test' : datasets.ImageFolder(test_dir, transform=data_transforms['validate'])}
    
    dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'validate' : torch.utils.data.DataLoader(image_datasets['validate'], batch_size=64),
                   'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)}
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx


def process_image(image_path):
    ''' 
    Scale, crop, normalize and reorder dimentions of image.
    
    Returns: Numpy array
    '''
    
    im = Image.open(image_path)
    
    # Resize image with shortest side 256
    if im.size[0] > im.size[1]:
        im.thumbnail((1000,256))
    else:
        im.thumbnail((256,1000))
    
    # Center crop to 224x224
    left = (im.width - 224)/2
    upper = (im.height - 224)/2
    right = left + 224
    lower = upper + 224
    im = im.crop((left, upper, right, lower))
    
    # Normalize
    np_im = np.array(im)/255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - means)/stds
    
    # Reorder dimensions, third to first with same order of other 2
    np_im = np_im.transpose((2,0,1))
    
    return np_im


def cat_to_name(json_path, predicted_cat):
    '''
    Uses json file to change predicted categories to 
    predicted names
    
    Returns: Predicted names
    '''
    
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    predicted_name = [cat_to_name[cat] for cat in predicted_cat]
    return predicted_name 


import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
import json
import os
from scipy.io import loadmat


def move_images(image_dir, image_labels):
    """
    Download at http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
        image_dir: 'Dataset images'
        image_labels: 'The image labels'

    Moves images:
        From: image_dir
        To: flowers/train, flowers/valid and flowers/test directory (ratio: 5/7, 1/7, 1/7).
            Subdirectories labeled with category (1-102).
            Each category is different species of flower.

    """

    if image_dir[-1] != '/':
        image_dir = image_dir + '/'

    # Make training, validation and testing directories
    os.mkdir('flowers')
    train_dir = 'flowers/train/'
    test_dir = 'flowers/test/'
    valid_dir = 'flowers/valid/'
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(valid_dir)

    # Make subdirectories
    for label in range(1, 103):
        os.mkdir(train_dir + '/' + str(label))
        os.mkdir(test_dir + '/' + str(label))
        os.mkdir(valid_dir + '/' + str(label))

    # Load labels (image index to category)
    labels = loadmat(image_labels, squeeze_me=True)
    labels_np = np.array(labels['labels'])

    # Move images to new subdirectories
    last_l = 0
    for i, l in enumerate(labels_np):
        if last_l != l:
            count = 1
            last_l = l
        else:
            count += 1

        label = str(l)
        i_str = str(i + 1)
        if len(i_str) == 1:
            src_start = image_dir + 'image_0000'
            dst_start = '/image_0000'
        elif len(i_str) == 2:
            src_start = image_dir + 'image_000'
            dst_start = '/image_000'
        elif len(i_str) == 3:
            src_start = image_dir + 'image_00'
            dst_start = '/image_00'
        else:
            src_start = image_dir + 'image_0'
            dst_start = '/image_0'

        src = src_start + i_str + '.jpg'
        if count % 6 == 0:
            dst = valid_dir + label + dst_start + i_str + '.jpg'
        elif count % 7 == 0:
            dst = test_dir + label + dst_start + i_str + '.jpg'
        else:
            dst = train_dir + label + dst_start + i_str + '.jpg'

        os.rename(src, dst)
    os.rmdir(image_dir)


def load_data(data_dir):
    """
    Load dataloaders with train, validation and test data.
    Transform data into 224x224 torch tensor and normalize to use with pretrained model.
    Perform augmentation on training data including random rotation, random resize and random horizontal flip.
    Get class_to_idx dictionary.
    
    Returns dataloaders and class_to_idx 
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'validate': transforms.Compose([transforms.Resize(255),
                                                       transforms.CenterCrop(224),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                            [0.229, 0.224, 0.225])])}
    
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'validate': datasets.ImageFolder(valid_dir, transform=data_transforms['validate']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['validate'])}
    
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'validate': torch.utils.data.DataLoader(image_datasets['validate'], batch_size=64),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)}
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx


def process_image(image_path):
    """
    Scale, crop, normalize and reorder dimentions of image.
    
    Returns: Numpy array
    """
    
    im = Image.open(image_path)
    
    # Resize image with shortest side 256
    if im.size[0] > im.size[1]:
        im.thumbnail((1000, 256))
    else:
        im.thumbnail((256, 1000))
    
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
    np_im = np_im.transpose((2, 0, 1))
    
    return np_im


def cat_to_name(json_path, predicted_cat):
    """
    Uses json file to change predicted categories to 
    predicted names
    
    Returns: Predicted names
    """
    
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    predicted_name = [cat_to_name[cat] for cat in predicted_cat]
    return predicted_name 


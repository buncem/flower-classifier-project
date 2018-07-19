import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
import utility

def make_classifier(input_units, hidden_units, dropout):
    '''
    Defines custom classifier
    
    Returns: classifier
    '''
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def make_model(base_model, hidden_units, dropout, learn_rate):
    '''
    Loads base model and freezes parameters so we don't train them.
    Replaces classifier with custom classifier.
    Defines loss function and optimizer.
    
    Returns: model, criterion, optimizer 
    '''
    
    if base_model in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet', 'densenet121', 'densenet169',
                      'densenet161', 'densenet201']:
        model = getattr(models, base_model)(pretrained=True)
        try:
            input_units=model.classifier[0].in_features
        except (AttributeError):
            input_units=model.classifier[1].in_features
        except (TypeError):
            input_units=model.classifier.in_features
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        classifier = make_classifier(input_units, hidden_units, dropout)
        model.classifier = classifier

        # Define loss function and Optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    return model, criterion, optimizer

def validate_model(model, testloader, criterion):
    '''
    Calculates average loss per batch and average accuracy for testloader
    
    Returns: average loss and accuracy
    '''
    
    model.eval()
    test_loss = 0
    accuracy = 0
    steps = 0
    for inputs, labels in testloader:

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        steps += 1
    return test_loss/steps, float(accuracy) * 100/steps

def train_model(model, criterion, optimizer, training_dataloader, validation_dataloader, epochs, gpu):
    '''
    Train model
    
    Returns: Trained model
    '''
    if gpu:
        model.to('cuda')
       
    print_every = 20
    running_loss = 0
    for e in range(epochs):
        model.train()
        steps = 0
        for ii, (inputs, labels) in enumerate(training_dataloader):
            steps += 1

            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
        loss, accuracy = validate_model(model, validation_dataloader, criterion)
        print('Average loss per validation batch: {0:.2f}'.format(loss))
        print('Accuracy on validation images: {0:.2f}'.format(round(accuracy,2)))
        
    return model
        
def save_model(model, dir_path, base_model, hidden_units, dropout, learn_rate):
    '''
    Save checkpoint
    '''
    
    checkpoint = {'base_model': base_model,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learn_rate': learn_rate,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, dir_path + '/checkpoint.pth')
    
def load_model(checkpoint_path, gpu):
    '''
    Loads vgg19 pretrained model, replaces classifier with custom classifier,
    loads model_state_dict from checkpoint file, defines loss criterion and optimizer,
    makes index to category dictionary.
    
    Returns: model, criterion, optimizer, idx_to_cat
    '''
    
    if gpu:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    base_model = checkpoint['base_model']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    learn_rate = checkpoint['learn_rate']
    
    # Load base model and attach custom classifier
    model, criterion, optimizer = make_model(base_model, hidden_units, dropout, learn_rate)
    
    # Load the model_state_dict from checkpoint file
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make dictionary for index to category
    cat_to_idx = checkpoint['class_to_idx']
    idx_to_cat = {value:key for key, value in cat_to_idx.items()}
    
    return model, criterion, optimizer, idx_to_cat

def predict(image_path, model, idx_to_cat, topk, gpu):
    ''' 
    Predict probabilities of top K categories.
    
    Returns: predicted probabilities and predicted categories
    '''
    
    # Process image and make into torch tensor of correct shape
    np_im = utility.process_image(image_path)
    torch_im = torch.tensor(np_im, dtype=torch.float).unsqueeze(0)
    
    # If using gpu model and image are in correct format
    if gpu:
        model.to('cuda')
        torch_im = torch_im.to('cuda')
    
    # Put model in eval mode (no dropout)
    model.eval()
    
    # Perform forward pass without keeping track of gradients
    with torch.no_grad():
        output = model.forward(torch_im)
    
    if not topk:
        topk = 1
    
    # Get top K categories and probabilities
    predicted_prob, predicted_idx = torch.exp(output).data.topk(topk)  
    predicted_prob = (predicted_prob.cpu().numpy().reshape((topk)) * 100).round(0)
    predicted_cat = [idx_to_cat[idx] for idx in predicted_idx.cpu().numpy().reshape((topk))]
    
    return predicted_prob, predicted_cat
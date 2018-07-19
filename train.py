import argparse
import numpy as np
import utility
import helper


parser = argparse.ArgumentParser(
    description='Use pretrained model for features and train custom classifier to predict flower species.')
parser.add_argument('data_directory', action="store",
                    help='store data directory path')
parser.add_argument('--save_dir',
                    dest='save_dir',
                    help='set directory to save checkpoint')
parser.add_argument('--arch',
                    default='vgg19',
                    choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet', 'densenet121',
                             'densenet169', 'densenet161', 'densenet201'],
                    dest='arch',
                    help='pick architecture (default: %(default)s)')
parser.add_argument('--learning_rate', type=float,
                    default=0.001,
                    dest='learning_rate',
                    help='pick learning rate (default: %(default)s)')
parser.add_argument('--hidden_units', type=int,
                    default=512,
                    dest='hidden_units',
                    help='pick number of hidden_units in classifier (default: %(default)s)')
parser.add_argument('--dropout', type=float,
                    default=0.5,
                    dest='dropout',
                    help='pick dropout (default: %(default)s)')
parser.add_argument('--epochs', type=int,
                    default=1,
                    dest='epochs',
                    help='pick number of epochs (default: %(default)s)')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu_t',
                    help='set gpu to true')
args = parser.parse_args()

# Load and preprocess data
dataloaders, class_to_idx = utility.load_data(args.data_directory)

# Load pretrained model and add custom classifier
model, criterion, optimizer = helper.make_model(args.arch, hidden_units=args.hidden_units, dropout=args.dropout,                                                                          learn_rate=args.learning_rate)

# Add attribute to model for category to index lookup
model.class_to_idx = class_to_idx

# Train model
model = helper.train_model(model, criterion, optimizer, dataloaders['train'], dataloaders['validate'], args.epochs, True)

# Save Model
if args.save_dir:
    helper.save_model(model, args.save_dir, args.arch, args.hidden_units, args.dropout, args.learning_rate)
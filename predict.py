import argparse
import numpy as np
import utility
import helper

parser = argparse.ArgumentParser(
    description='Predict category or name of flower in image')
parser.add_argument('image_path', action="store",
                    help='image path')
parser.add_argument('checkpoint_path', action="store",
                    help='checkpoint path')
parser.add_argument('--top_k', type=int,
                    dest='top_k',
                    help='return top k most likely categories or names')
parser.add_argument('--category_names',
                    dest='category_names',
                    help='file path that maps category to real names for flowers')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu_t',
                    help='set gpu to true')
args = parser.parse_args()

# Load model
model, criterion, optimizer, idx_to_cat = helper.load_model(args.checkpoint_path, args.gpu_t)

# Predict image using model
predicted_prob, predicted_cat = helper.predict(args.image_path, model, idx_to_cat, args.top_k, args.gpu_t)

if not args.top_k and not args.category_names:
    print("The flower category is {} with {}% probability.".format(predicted_cat[0], predicted_prob[0]))
elif args.top_k and not args.category_names:
    print("The {} most likely flower categories are {} with {} probabilities.".format(args.top_k, predicted_cat, predicted_prob))
elif not args.top_k and args.category_names:
    predicted_name = utility.cat_to_name(args.category_names, predicted_cat)
    print("The flower name is {} with {}% probability.".format(predicted_name[0], predicted_prob[0]))
else:
    predicted_name = utility.cat_to_name(args.category_names, predicted_cat)
    print("The {} most likely flower names are {} with {} probabilities.".format(args.top_k, predicted_name, predicted_prob))
                    
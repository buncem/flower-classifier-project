import argparse
import utility
import helper

parser = argparse.ArgumentParser(
    description='Move image downloads to newly created flowers directory,'
                ' delete download directory.')
parser.add_argument('image_dir', action="store",
                    help='image directory path')
parser.add_argument('labels_file', action="store",
                    help='labels file path')
args = parser.parse_args()

print('Moving files..')
utility.move_images(args.image_dir, args.labels_file)
print('flowers directory created')
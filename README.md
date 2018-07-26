Flower Classifier Project
------------------------------

Image classifier for 102 species of flowers

Installation
--------------------

### Environment requirements

* Create a conda environment with python=3.6 and the following packages.
    * conda create --name myenv python=3.6 numpy cython jupyter nb_conda_kernels scipy
    * source activate myenv
* Go to [PyTorch website](https://pytorch.org/) and follow directions to install PyTorch for your system.

### Clone requirements

* If not installed please install [Git LFS](https://git-lfs.github.com).
    * Checkout this [Tutorial](https://www.atlassian.com/git/tutorials/git-lfs) with more details on how Git LFS works and how to install it.
* Clone repository
    * git clone https://github.com/buncem/flower-classifier-project.git

### Two ways to use the app

* Use checkpoint.pth included in the repository to predict flower types.
    * Doesn't require downloading the Dataset.
    * Doesn't require using a GPU to train.
    * Test photos included in repository under test-photos directory.

* Train classifier yourself.
    * GPU is highly recommended to train in a reasonable time.
    * Downloading training data is required, see "Download data" section


### Download data

* Download data into main repository from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
    * Under downloads download "Dataset images" and "The image labels"
    * Untar 102flowers.tar which should create a jpg directory.
    * At this point you should have jpg directory with 8189 jpgs in it and a imagelabels.mat file

### Make Training, Validation and Testing sets

* Run setup.py script.
    * python setup.py -h
        * Usage help
    * python setup.py jpg imagelabels.mat
        * Creates flowers directory with train, valid and test subdirectories.
        * Each subdirectory has subdirectories labeled 1-102 with jpg files corresponding to 102 flower labels.
            * Note that cat_to_name.json has mappings of numeric labels to actual names of flowers.

# Project Code for Udacity's AI Programming with Python Nanodegree Program.


# The First Project Builds a Dog Breed Classifier

## Dog Classifier Project Instructions

### Principal Objectives

1. Correctly identify which pet images are of dogs (even if the breed is misclassified) and which pet images aren't of dogs.
2. Correctly classify the breed of dog, for the images that are of dogs.
3. Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve objectives 1 and 2.
4. Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms takes to run.

### TODO:
Edit program check_images.py

The check_images.py is the program file that you will be editing to achieve the four objectives above. 

This file contains a main() function that outlines how to complete this program through using functions that have 
not yet been defined. 

You will be creating these undefined functions in check_images.py to achieve the objectives above.

All of the TODOs are listed in check_images.py.

# The Second Project Builds a Flower Image Classifier

In this project code is developed for an image classifier built with PyTorch.

The code for the image classifier is developed in a jupyter notebook:  Image_Classifier_Project.ipynb.

Then the code is adapted and converted into a command line application.  That is done in the two files:  train.py and predict.py

# Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.

## 1. Train

Train a new network on a data set with train.py

    - Basic usage: python train.py data_directory
    - Prints out training loss, validation loss, and validation accuracy as the network trains
    - Options: * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory * Choose architecture: python train.py data_dir --arch "vgg13" * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
    - Use GPU for training: python train.py data_dir --gpu

## 2. Predict

Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    - Basic usage: python predict.py /path/to/image checkpoint
    - Options: * Return top KK most likely classes: python predict.py input checkpoint --top_k 3 * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
    - Use GPU for inference: python predict.py input checkpoint --gpu

# CS 203 Assignment 5
## Overview
In this assignment, we implemented and trained a ResNet50 model with augmented and non-augmented datasets and compared them on test dataset. Our objective is to improve metrics on test dataset by applying transformations to the train dataset. We documented our approach, dataset details, preprocessing steps, training methodology, and evaluation metrics in the jupyter notebook.

## Prerequisites
To run this project, we require the following:
- PyTorch, TensorFlow and Keras
- NumPy, Matplotlib, and other supporting libraries
- GPU (recommended) for efficient training

## Dataset
We utilize an image dataset suitable for classification tasks. The dataset undergoes augmentation techniques such as flipping, rotation, brightness and other adjustments to enhance diversity and improve model robustness.

## Data Preprocessing
We preprocessed the dataset by splitting it into training and test sets. Augmentations were applied to ensure variability.

## Model Architecture
We employed the ResNet50 architecture, a deep convolutional neural network. The model is initialized with pre-trained weights and fine-tuned for our specific classification task.

## Training Process
The training process involves:
1. Loading and preprocessing the dataset.
2. Applying data augmentation to the training set.
3. Configuring the model with an appropriate optimizer and loss function.
4. Training the model with batch processing.

## Evaluation Metrics
We assess the model's performance on augmented and non-augmented data using:
- Accuracy
- Loss
- Confusion matrix
- Precision, Recall, and F1-score
#### on test dataset

## How to Run
To execute the project, follow these steps:
1. Install the required dependencies.
2. Load the dataset and preprocess it.
3. Train the model using the provided training script.
4. Evaluate the trained model using test data.
5. Use the trained model for inference on new images.

## Contributors
This assignment is a collaborative effort by Jeet Joshi (23110148) and Kain Harshil Shivkumar (23110151).

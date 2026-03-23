# Internship Task 1: MNIST Image Classification

## Project Overview
This repository contains the solution for the internship Task 1 test. It implements a unified interface, a wrapper class, and 3 classes for RF, CNN, and FFNN models for MNIST image classification. The demonstration notebook contains examples of how every model works and edge cases descriptions.

## Project Structure
src/:
  1) 'Convolutional_NN_module.py' - contains the ConvolutionalNN model class, powered by Keras.
  2) 'FFNN_module.py' - contains the FeedForwardNN model class, powered by Keras.
  3) 'RandomForest_module.py' - contains the RandomForest baseline model, powered by sklearn.
  4) 'MnistClassifier_module.py' - wrapper class / factory pattern, implements algorithm switch.
  5) 'MnistClassifierInterface_module.py' - unified interface with train and predict abstract methods.

notebooks/: 'Inference_demo_notebook.ipynb' - notebook demonstrating all models' performance and edge cases descriptions.

## Edge Cases Investigated
Analysis of out-of-distribution behavior, NaN values propagation, and unexpected shape input cases.

## Installation & Setup
1. Clone the repository:
   git clone https://github.com/vozniukivan0501-cmyk/internship_task1

2. Install dependencies:
   pip install -r requirements.txt

## Usage
Example of how to use the unified wrapper class:

from src.MnistClassifier_module import MnistClassifier

# Initialize model (options: 'rf', 'nn', 'cnn')
classifier = MnistClassifier('cnn')

# Train and predict
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)

# Feedforward Neural Network from Scratch

##### This project implements a Feedforward Neural Network (FNN) from scratch using only Python and NumPy, without relying on deep learning libraries like TensorFlow or PyTorch. The network is trained on the MNIST dataset to classify handwritten digits (0-9).

## Features

Custom Activation Functions: Implements ReLU, sigmoid, and softmax.
Training and Evaluation: Includes forward propagation, backpropagation, and gradient descent.
Custom Loss Functions: Uses cross-entropy loss.
Performance: Achieves an accuracy of ~79% on the MNIST test set.
Dataset

### The MNIST dataset is used, which consists of:

60,000 training examples
10,000 test examples
Images are 28x28 pixels in grayscale.
## Requirements

Python 3.7+
NumPy
Keras (only for loading the MNIST dataset)
Getting Started

### Clone the repository:
git clone https://github.com/your-username/scratchnn.git
cd scratchnn
### Install dependencies:
pip install numpy keras
Run the notebook:
Use Jupyter Notebook or any compatible editor to open and execute scratchnn.ipynb.


## How It Works

#### Data Preprocessing:
Normalizes pixel values to a range of [0, 1].
Converts labels to one-hot encoding.
#### Neural Network Architecture:
Input Layer: 28x28 = 784 neurons (flattened image)
Hidden Layer: 64 neurons with ReLU activation
Output Layer: 10 neurons with softmax activation
#### Training:
Uses forward propagation to calculate predictions.
Backpropagation is used to compute gradients.
Parameters are updated using gradient descent.
#### Evaluation:
Outputs accuracy and loss for each epoch during training.
Results

The model achieved a peak accuracy of ~79.38% on the MNIST test set, demonstrating the feasibility of building an FNN from scratch.

## Future Improvements

Implement additional optimizers like Adam or RMSProp.
Add regularization techniques to improve generalization.
Experiment with deeper architectures.


## Contributing

Feel free to fork this repository and submit pull requests for improvements or additional features.

# MNIST Convolutional Neural Network

This project is my first convolutional neural network using TensorFlow! It classifies handwritten digits from the MNIST dataset. Big thanks to [DeepLearning.ai](https://www.coursera.org/learn/introduction-tensorflow) for their amazing course!

## Features
- Utilizes Conv2D layers for image classification
- Implements early stopping once 99.5% accuracy is achieved
- Achieves high accuracy on training data

## Requirements
- Python 3.x
- TensorFlow

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/reese8272/mnist-cnn-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mnist-cnn-project
   ```
3. Install TensorFlow:
   ```bash
   pip install tensorflow
   ```

## How to Run
Run the script using Python:
```bash
python MNistTest.py
```

## How It Works
1. The script loads the MNIST dataset.
2. Images are reshaped and normalized for optimal model performance.
3. A convolutional neural network is defined with the following architecture:
   - Two Conv2D layers followed by MaxPooling2D layers
   - Flatten layer to convert 2D features into 1D
   - Dense layer with 128 units and ReLU activation
   - Output Dense layer with 10 units and softmax activation for classification
4. Early stopping is implemented to halt training once the model achieves 99.5% accuracy.

## Results
The model is designed to stop training once it hits 99.5% accuracy on the training set. This ensures efficient use of resources and prevents overfitting.

## Acknowledgments
- Thanks to [DeepLearning.ai](https://www.coursera.org/learn/introduction-tensorflow) for their course that guided the creation of this project.

## License
This project is for educational purposes.

---
Happy coding! 🚀


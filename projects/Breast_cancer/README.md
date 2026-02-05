Breast Cancer Diagnosis using Logistic Regression
Project Overview
This project implements a logistic regression model from scratch to diagnose breast cancer as malignant (M) or benign (B) based on cell nucleus characteristics from the Wisconsin Breast Cancer Dataset. The model achieves 88.37% accuracy on the test set.

Dataset
Source: Wisconsin Breast Cancer Dataset

Samples: 569 instances

Features: 30 numerical features describing cell nucleus characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)

Target Variable: Diagnosis (M = 1, B = 0)

Project Structure
The implementation consists of the following key components:

1. Data Preprocessing
Removed unnecessary columns ('id' and 'Unnamed: 32')

Encoded diagnosis labels: 'M' → 1, 'B' → 0

Normalized features using min-max scaling

Split data into training (85%) and testing (15%) sets

2. Logistic Regression Implementation
The model is built from scratch with the following functions:

initialize_weights_and_bias(): Initializes weights and bias with small random values

sigmoid(): Implements the sigmoid activation function

forward_backward_propagation(): Performs forward pass and calculates gradients

update(): Updates weights and bias using gradient descent

predict(): Makes predictions on test/train data

logistic_regression(): Main function to train and evaluate the model

3. Model Training
Learning Rate: 0.01

Iterations: 1000

Cost Function: Binary cross-entropy

Optimization: Gradient Descent

Results
Training Accuracy: 90.68%

Testing Accuracy: 88.37%

Cost Reduction: Decreased from 0.6928 to 0.5156 over 1000 iterations

Dependencies
Python 3.x

NumPy

pandas

scikit-learn

matplotlib

How to Run
Ensure all dependencies are installed

Place the data.csv file in the project directory

Run the code sequentially in the Jupyter notebook

The model will output training progress and final accuracy scores

Key Features
✅ Implementation of logistic regression from scratch

✅ Proper data preprocessing and normalization

✅ Cost tracking during training

✅ Performance evaluation on test set

✅ Clean, well-documented code

Potential Improvements
Implement regularization to prevent overfitting

Add cross-validation for better hyperparameter tuning

Experiment with different optimization algorithms (e.g., Adam)

Add feature selection to reduce dimensionality

Implement k-fold cross-validation for more robust evaluation

Author
This project demonstrates fundamental machine learning concepts by implementing logistic regression from scratch for a binary classification task.

License
This project is for educational purposes. The dataset is publicly available for research use

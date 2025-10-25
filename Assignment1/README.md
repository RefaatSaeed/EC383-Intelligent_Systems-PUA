# Assignment 1 – Linear Regression using Stochastic Gradient Descent (SGD)
**Course:** EC383 – Intelligent Systems  
**University:** PUA  

This assignment implements Linear Regression from scratch using Stochastic Gradient Descent (SGD) in Python, without using any pre-built machine learning libraries. The goal is to understand how the model learns by updating weights and bias during training.

Dataset used:
| X1 | X2 | X3 | Label |
|----|----|----|--------|
| 73 | 80 | 75 | 152 |
| 93 | 88 | 93 | 185 |
| 89 | 91 | 90 | 180 |
| 96 | 98 | 100 | 196 |
| 73 | 66 | 70 | 142 |
| 53 | 46 | 55 | 101 |

Features: X1, X2, X3  
Target: Label  

Steps:
1. Load the dataset from CSV and rename columns to x, y, z, and label.  
2. Normalize the data using X = X / np.max(X, axis=0) and y = y / np.max(y).  
3. Initialize weights and bias.  
4. Train the model using Stochastic Gradient Descent:  
   y_pred = np.dot(X[i], weights) + b  
   error = y_pred - y[i]  
   weights = weights - lr * error * X[i]  
   b = b - lr * error  
5. Compute the Mean Squared Error (MSE) after training.

Example Output:
Trained weights: [0.45 0.43 0.47]  
Bias: 0.02  
MSE: 0.002  

What I Learned:
- How Linear Regression works mathematically.  
- How gradient descent updates weights to minimize loss.  
- The role of normalization in stable learning.  
- How to calculate and interpret Mean Squared Error (MSE).  

Files:
Assignment1/
  ├── Linear_SGD.ipynb     (Jupyter Notebook)
  ├── data.csv             (Dataset)
  └── README.md            (This file)

How to Run:
1. Open Linear_SGD.ipynb in Jupyter Notebook.  
2. Run all cells in order.  
3. Check the final trained weights, bias, and MSE value.


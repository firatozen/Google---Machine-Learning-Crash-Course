# Google - Machine Learning Crash Course
# Machine Learning Crash Course by Google - Programming Exercises

This repository contains solutions and implementations of programming exercises from Google's **Machine Learning Crash Course (MLCC)**. The course provides a fast-paced, practical introduction to machine learning, featuring video lectures, interactive visualizations, and hands-on practice. It covers fundamental and advanced machine learning concepts, including regression, classification, handling numerical data, and addressing fairness and bias in machine learning systems.

## Course Overview

The **Machine Learning Crash Course** is designed for individuals looking to understand the basics of machine learning. It is a great starting point for anyone interested in learning how to apply machine learning techniques to real-world problems.

- **Course Duration**: ~15 hours
- **Modules**: 12 modules covering a range of machine learning topics
- **Hands-on Exercises**: 100+ exercises

The following topics are included in the course:
- **ML Models**: Linear regression, logistic regression, and binary classification.
- **Data Handling**: Working with numerical and categorical data, generalization, and overfitting.
- **Advanced ML Models**: Introduction to neural networks, embeddings, and large language models.
- **Real-World ML**: Best practices for production ML systems, AutoML, and ML fairness.

## Programming Exercises

The following programming exercises were completed as part of this course:

### 1. **Linear Regression**

In this exercise, I implemented a **linear regression** model to predict continuous values based on input features. Linear regression is one of the simplest and most commonly used algorithms for regression tasks. The goal is to find the best-fitting line that minimizes the difference between predicted and actual values. The exercise covers:
- Linear models
- Loss function
- Gradient descent
- Hyperparameter tuning

**File**: `linear_regression.ipynb`  
**Explanation**: This notebook includes the implementation of a linear regression model using the gradient descent optimization method to minimize the loss function.

### 2. **Classification**

This exercise involved implementing a **logistic regression** model for binary classification. The model predicts the probability of a given outcome (e.g., success or failure). Logistic regression uses a sigmoid function to output values between 0 and 1, which can be interpreted as probabilities.

**File**: `classification.ipynb`  
**Explanation**: The notebook demonstrates the use of logistic regression, evaluates performance with confusion matrices, and calculates accuracy, precision, recall, and AUC.

### 3. **Working with Numerical Data**

This exercise focuses on the techniques used to handle and preprocess **numerical data** effectively. The primary tasks include analyzing, transforming, and cleaning the data. Specific challenges addressed include:
- Handling missing or incorrect values
- Scaling and normalizing numerical features

**File**: `Numerical data (Programming Statistics and bad values).ipynb`  
**Explanation**: This notebook covers common transformations like data imputation and feature scaling to prepare data for training machine learning models.

### 4. **Fairness in Machine Learning**

In this exercise, I explored how to detect and address **bias** and **fairness** issues in machine learning models. ML models can inadvertently reflect biases present in the data, which can lead to unfair or inaccurate predictions. The exercise covers:
- Identifying bias in data
- Auditing models for fairness
- Implementing techniques to mitigate bias

**File**: `Fairness.ipynb`  
**Explanation**: This notebook implements fairness audits for machine learning models, including strategies to identify and mitigate bias in predictions.

## Repository Structure

- `linear_regression.ipynb`: Code implementation of linear regression for continuous prediction tasks.
- `classification.ipynb`: Code implementation of logistic regression for binary classification tasks.
- `Numerical data (Programming Statistics and bad values).ipynb`: Code handling the transformation of numerical data, including imputation and scaling.
- `Fairness.ipynb`: Code to identify and mitigate bias and fairness issues in machine learning models.

## Getting Started

To run the exercises and test the models, follow the steps below:

### Prerequisites

- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `jupyter`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ml-crash-course-exercises.git

# Diabetes Prediction

A machine learning project that uses various classification algorithms to predict diabetes based on diagnostic measurements.

## Overview

This project uses multiple classification models to predict diabetes outcomes using the Pima Indians Diabetes dataset. The code demonstrates how to prepare data, train different models, and evaluate their performance.

## Features

- Data loading and preprocessing
- Train/test splitting
- Feature standardization
- Multiple classification algorithms:
  - Support Vector Machine (SVC)
  - Logistic Regression
  - Random Forest
  - LazyClassifier for automated model comparison
- Hyperparameter tuning with GridSearchCV
- Model evaluation metrics (accuracy, precision, recall, F1-score)
- Model saving functionality

## Requirements

- Python 3.x
- pandas
- scikit-learn
- lazypredict
- pickle

## Usage

1. Place your "diabetes.csv" file in the same directory as the script
2. Run the script to:
   - Load and preprocess the dataset
   - Split data into training and testing sets
   - Standardize features
   - Train and compare multiple classifier models
3. View the performance comparison of different models

## Notes

The code includes commented-out sections for:
- Data exploration and profiling with ydata_profiling
- Individual classifier implementations
- Hyperparameter tuning with GridSearchCV
- Detailed evaluation metrics
- Model saving

Uncomment these sections as needed for your specific analysis requirements.

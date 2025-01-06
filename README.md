# Diabetes Prediction Using Perceptron

This project focuses on using a Perceptron model for diabetes prediction. The study was conducted during my Master's program (The University of Adelaide) and demonstrates machine learning, data preprocessing, and performance evaluation expertise.

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)
- [Installation](#installation)
- [Methodology](#methodology)  
- [Implementation](#implementation)  
- [Results](#results)  
- [Conclusion](#conclusion)  
- [Future Scope](#future-scope)  

## Overview

This project aimed to predict diabetes in patients using a Perceptron-based model. The project emphasizes applying machine learning techniques to classify patients as diabetic or non-diabetic based on their medical data.

## Dataset

The dataset used for this project is the Pima Indians Diabetes Database, sourced from the UCI Machine Learning Repository. It contains data on 768 patients, with the following features:

- **Pregnancies:** Number of times the patient was pregnant.  
- **Glucose:** Plasma glucose concentration after 2 hours in an oral glucose tolerance test.  
- **Blood pressure:** Diastolic blood pressure (mm Hg).  
- **SkinThickness:** Triceps skin fold thickness (mm).  
- **Insulin:** 2-Hour serum insulin (mu U/ml).  
- **BMI:** Body Mass Index (weight in kg/(height in m)^2).  
- **DiabetesPedigreeFunction:** A function that scores the likelihood of diabetes based on family history.  
- **Age:** Age of the patient (years).  
- **Outcome:** Binary target variable (1 = diabetic, 0 = non-diabetic).

## Installation
To run this project locally, ensure you have the following dependencies installed:

Python 3.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
TensorFlow
You can install the required packages using pip:

pip install numpy pandas scikit-learn matplotlib seaborn tensorflow


## Methodology

1. **Data Preprocessing:**  
   - Handled missing values using mean/mode imputation.  
   - Normalized features to standardize the scale.  

2. **Model Design:**  
   - A single-layer Perceptron was implemented for binary classification.  
   - The model used the sigmoid activation function for predicting probabilities.

3. **Performance Evaluation:**  
   - Evaluated using metrics such as accuracy, precision, recall, F1-score, and Area Under the Curve (AUC).  
   - Utilized k-fold cross-validation to ensure robust results.

4. **Visualization:**  
   - Data distribution is visualized using histograms and pair plots.  
   - Model performance metrics visualized using confusion matrices and ROC curves.

## Implementation

The project was implemented in Python using the following libraries:  
- **NumPy:** For numerical computations.  
- **Pandas:** For data manipulation.  
- **Matplotlib and Seaborn:** For data visualization.  
- **Scikit-learn:** For model building and evaluation.  

The Perceptron model was implemented using Scikit-learn's `Perceptron` class with the following steps:  
1. Splitting the dataset into training and testing subsets.  
2. Training the model on the training set.  
3. Evaluating the model on the test set and reporting metrics.  

## Results

- **Accuracy:** Approximately 66.23% on the test set with bias(after experimenting with bias and without bias) and validation accuracy as 78.8%.  
- **Precision, Recall, and F1-Score:** Demonstrated strong predictive capability.  
- **ROC-AUC Score:** Indicated the model's ability to distinguish between diabetic and non-diabetic patients effectively.  

## Conclusion

The Perceptron model achieved satisfactory results for diabetes prediction, demonstrating the potential of simple machine learning models in healthcare applications. The project highlights the importance of data preprocessing, feature normalization, and model evaluation in achieving reliable predictions.

## Future Scope

- Extend the model to include advanced architectures like Multi-Layer Perceptrons (MLPs).  
- Experiment with hyperparameter tuning to improve performance further.  
- Incorporate additional datasets for generalized results.  
- Develop a user-friendly interface for medical practitioners to input patient data and receive predictions in real time.

This project reflects the practical application of machine learning techniques in solving real-world problems and provides a foundation for future research in predictive healthcare analytics.


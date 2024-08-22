# Breast Cancer Classification
## Overview
This project focuses on the classification of breast cancer using machine learning techniques. The goal is to build a model that can accurately predict whether a tumor is benign or malignant based on the features provided in the Wisconsin Breast Cancer Dataset.

## Dataset
The dataset used in this project is the Wisconsin Breast Cancer Dataset, which includes 569 instances of tumors, each with 30 features such as mean radius, mean texture, and mean smoothness, among others. The target variable indicates whether the tumor is benign (label 0) or malignant (label 1).

## Project Structure
The project is structured as follows:

- data/: Contains the dataset used for the project.
- notebooks/: Jupyter notebooks that include data exploration, preprocessing, model training, and evaluation.
- models/: Saved models for future use or analysis.
- README.md: Overview of the project (this file).

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
`git clone https://github.com/mayankdobhal0025/breast-cancer-classification.git`
2. Navigate to the project directory:
`cd breast-cancer-classification`
3. Create a virtual environment:
`python -m venv venv`
4. Activate the virtual environment:
  - On Windows:
    `venv\Scripts\activate`
  - On macOS/Linux:
    `source venv/bin/activate`
5. Install the required dependencies:
`pip install -r requirements.txt`
## Usage
To run the classification model:
1. Preprocess the data
2. Train the model
3. Evaluate the model

## Models Used
The following machine learning models were explored in this project:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree
- The models were evaluated based on accuracy, precision, recall, and F1-score. Hyperparameter tuning was performed to optimize the model performance.

## Results
The best-performing model in this project achieved an accuracy of 99% on the test dataset. This model was selected based on its balance between precision and recall, ensuring both high accuracy and reliable predictions.

## Conclusion
This project demonstrates the effectiveness of machine learning models in classifying breast cancer. By leveraging various algorithms and feature engineering techniques, we were able to build a robust model that can assist in early detection and treatment decisions.

## Future Work
Future improvements could include:

Implementing more advanced models such as neural networks.
Exploring additional feature selection techniques to improve model performance.
Integrating the model into a web application for real-time predictions.
## Acknowledgments
The dataset used in this project was provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).
Thanks to the open-source community for providing the libraries and tools used in this project.

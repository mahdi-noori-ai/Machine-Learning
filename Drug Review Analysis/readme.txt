
---

# Drug Review Analysis

## Overview
This Jupyter notebook is designed for analyzing drug reviews from a dataset, with the primary objective of predicting a target variable (likely the rating of a drug) based on various features, including text reviews and other categorical and numerical data.

## Dataset
The notebook utilizes a drug review dataset fetched from the UCI Machine Learning Repository. The dataset contains features such as user reviews, drug names, conditions treated, and the rating provided by users.

## Task and Purpose
The main task is to build predictive models that estimate the rating of a drug based on features extracted from the dataset, including text reviews processed through natural language processing techniques like TF-IDF vectorization.

## Workflow
1. **Data Fetching**: The dataset is fetched from the UCI repository and loaded into a pandas DataFrame for further analysis.
2. **Feature Extraction**: The notebook extracts features from the dataset, including numerical, categorical, and textual data. Text data is vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert text reviews into numerical features.
3. **Data Preprocessing**: 
   - Text data is vectorized using TF-IDF.
   - Categorical variables are converted into dummy variables (one-hot encoding).
   - The target variable (e.g., drug rating) is separated from the features.
4. **Data Visualization**: The distribution of the target variable is visualized using histograms to understand its distribution.
5. **Data Splitting**: The dataset is split into training and testing sets to allow for model training and evaluation.
6. **Model Building**: 
   - A pipeline is created to standardize features, generate polynomial features, and apply regression models.
   - The notebook explores models like Lasso Regression and ElasticNet Regression.
7. **Model Evaluation**: The models are evaluated based on metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared to assess their performance.

## Dependencies
- Python 3.x
- Jupyter Notebook
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Numpy
- Ucimlrepo (for fetching the dataset)

## How to Run
1. Clone the repository or download the notebook.
2. Install the required dependencies using `pip`:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib numpy ucimlrepo
   ```
3. Run the notebook using Jupyter:
   ```bash
   jupyter notebook drug.ipynb
   ```
4. Follow the steps in the notebook to execute the analysis and view the results.

## Conclusion
This notebook provides a framework for analyzing drug reviews using a combination of text processing and machine learning techniques. It offers insights into how various features, including textual reviews, can be used to predict drug ratings, which could be valuable for pharmaceutical analysis and customer feedback interpretation.

---

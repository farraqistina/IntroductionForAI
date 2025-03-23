# IntroductionForAI

# README: AI Coursework

## Project Overview
This project focuses on two machine learning applications:
1. **Linear Regression for California Housing Prices**
2. **Human Activity Recognition using SVM (Support Vector Machines)**

Each section covers data preparation, model training, evaluation, and visualization to analyze and interpret the results.

---

## Running Instructions on a Local Machine
### Prerequisites
Before running the project, ensure you have **Python 3.x** installed.

### Installing Required Libraries
To install the necessary dependencies, run the following command:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
If using **Jupyter Notebook**, install it with:
```bash
pip install notebook
```

### Running the Code
1. **Clone or download** the project repository.
2. Open a terminal or command prompt and navigate to the project directory:
   ```bash
   cd path/to/project
   ```
3. **If using Jupyter Notebook**, start it by running:
   ```bash
   jupyter notebook
   ```
   Then open the relevant notebook file and execute the cells sequentially.
4. **If running Python scripts directly**, use:
   ```bash
   python uci_har_svm.py
   python filemname.py
   ```
5. **For the UCI HAR Dataset**, ensure that the dataset ZIP file is extracted into the correct directory before running the classification task.

---

## 1. Linear Regression: Predicting California Housing Prices
### Objective:
This section applies **Linear Regression** to predict housing prices based on median income.

### Dataset:
- **California Housing Dataset** (from `sklearn.datasets.fetch_california_housing`)
- Predictor: `MedInc` (Median income in block groups)
- Target: `MedHouseVal` (Median house value)

### Workflow:
1. Load and explore the dataset.
2. Standardize the feature using **StandardScaler** for consistency.
3. Train two models:
   - **Batch Gradient Descent (BGD)**: Using `SGDRegressor` with `learning_rate='constant'`.
   - **Stochastic Gradient Descent (SGD)**: Using `SGDRegressor` with `learning_rate='optimal'`.
4. Evaluate model performance:
   - **Mean Squared Error (MSE)**
   - **R² score**
5. Make predictions for a given income level (e.g., `$80,000`).
6. Generate visualizations:
   - Scatter plot of **actual vs. predicted house prices**
   - **Residual distribution plot**
   - **Box plot of errors**
   - **Comparison of R² scores**

---

## 2. Human Activity Recognition using SVM
### Objective:
This section classifies human activities as **Active (1) or Inactive (0)** using **Support Vector Machines (SVM)**.

### Dataset:
- **UCI HAR Dataset** (Human Activity Recognition Using Smartphones)
- Contains **561 sensor readings** from smartphone accelerometers and gyroscopes.
- Activity labels:
  - **Active (1)**: Walking, Walking Upstairs, Walking Downstairs
  - **Inactive (0)**: Sitting, Standing, Lying Down

### Workflow:
1. **Extract and load** the dataset.
2. Preprocess the data:
   - Assign correct feature names.
   - Map activity labels to binary classes.
   - Standardize sensor readings using **StandardScaler**.
   - Apply **Principal Component Analysis (PCA)** to reduce dimensions to **50 components**.
3. Train **SVM models** with different kernel types:
   - **Linear**
   - **Polynomial**
   - **Radial Basis Function (RBF)**
4. Evaluate model performance:
   - **Accuracy comparison**
   - **Confusion matrices**
   - **Classification reports**
5. Generate visualizations:
   - **Accuracy bar chart** comparing different kernels
   - **Confusion matrices for each SVM model**

---

## Requirements
- Python 3.x
- Jupyter Notebook (if running in a notebook environment)
- Required Libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```

---

## Evaluation Metrics
### Linear Regression:
- **Mean Squared Error (MSE)**: Measures how far predicted values are from actual prices.
- **R² Score**: Indicates how well the model explains the variability of house prices.

### SVM Classification:
- **Accuracy**: Measures the proportion of correctly classified activities.
- **Confusion Matrix**: Shows classification results for different activity types.
- **Classification Report**: Provides Precision, Recall, and F1-score.

---

## Additional Notes
- Before running the **UCI HAR Dataset** section, ensure that the dataset ZIP file is properly extracted.
- The project is structured to follow best practices in data preprocessing, model evaluation, and visualization.
- Both scripts can be run in Jupyter Notebook or directly as Python files.

---

### Author: Farra Qistina Binti Mohmmad Nasir
### Course: AI Coursework
### Institution: City, University of London


# Heart Disease Prediction using Gradient Boosting Classifier

This project aims to predict the presence of heart disease in patients based on 13 clinical attributes. The analysis involves exploratory data analysis (EDA) to understand the relationships between different features and the target variable, followed by the implementation and optimization of a Gradient Boosting Classifier model.

## Dataset

[Kaggle Link](https://www.kaggle.com/code/emirhanhasrc/eda-gradiantboosting-classifier)

The dataset used is the "Heart Disease" dataset, which is a subset of the larger Cleveland Clinic Foundation database from the UCI Machine Learning Repository. It contains 303 instances and 14 attributes, including the target variable.

### Dataset Features

The dataset consists of the following features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| **age** | Age of the patient in years | Numeric |
| **sex** | Sex of the patient (1 = male; 0 = female) | Numeric |
| **cp** | Chest Pain Type | Numeric |
| | 0: Typical Angina | |
| | 1: Atypical Angina | |
| | 2: Non-Anginal Pain | |
| | 3: Asymptomatic | |
| **trestbps** | Resting Blood Pressure (in mm Hg) | Numeric |
| **chol** | Serum Cholestoral in mg/dl | Numeric |
| **fbs** | Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false) | Numeric |
| **restecg**| Resting Electrocardiographic Results | Numeric |
| | 0: Normal | |
| | 1: Having ST-T wave abnormality | |
| | 2: Showing probable or definite left ventricular hypertrophy| |
| **thalach**| Maximum Heart Rate Achieved | Numeric |
| **exang** | Exercise Induced Angina (1 = yes; 0 = no) | Numeric |
| **oldpeak**| ST depression induced by exercise relative to rest | Numeric |
| **slope** | The slope of the peak exercise ST segment | Numeric |
| **ca** | Number of major vessels (0-3) colored by flourosopy | Numeric |
| **thal** | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect) | Numeric |
| **target** | Heart Disease (0 = no; 1 = yes) | Numeric |

## Exploratory Data Analysis (EDA)

A thorough EDA was conducted to uncover insights and patterns within the data.

1.  **Correlation Matrix**: A heatmap was generated to visualize the correlation between all features. This helps in understanding the relationships between variables. The 'cp' (chest pain type), 'thalach' (maximum heart rate achieved), and 'slope' features show a positive correlation with the target variable, while 'exang', 'oldpeak', 'ca', and 'thal' show a negative correlation.
2.  **Data Distribution**: Histograms for all variables were plotted to understand their distributions. The age distribution is fairly normal, centered around the late 50s.
3.  **Target Variable Distribution**: A pie chart revealed that the dataset is well-balanced, with **54.5%** of patients having heart disease and **45.5%** not having the disease.
4.  **Age vs. Target**: A boxenplot showed that the median age for patients with heart disease is slightly lower than for those without.
5.  **Sex vs. Target**: A countplot indicated that a higher proportion of males in the dataset have heart disease compared to females.
6.  **Chest Pain Type vs. Target**: It was observed that patients experiencing 'Typical Angina' are less likely to have heart disease, while those with 'Atypical Angina' and 'Non-Anginal Pain' are more likely to have it.

## Modeling and Evaluation

A Gradient Boosting Classifier was chosen for this classification task due to its high predictive power.

### 1. Data Preprocessing

-   The dataset was split into features (X) and the target variable (y).
-   It was then divided into a training set (80%) and a testing set (20%).
-   A check for multicollinearity was performed, and no features had a correlation coefficient above the threshold of 0.85, so no features were dropped.

### 2. Baseline Model

A standard `GradientBoostingClassifier` was trained on the training data. The model's performance was evaluated on the test set, achieving the following results:

| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Healthy)** | 0.74 | 0.82 | 0.78 | 28 |
| **1 (Heart Disease)** | 0.83 | 0.76 | 0.79 | 33 |
| **accuracy** | | | **0.79** | **61** |
| **macro avg** | 0.79 | 0.79 | 0.79 | 61 |
| **weighted avg** | 0.79 | 0.79 | 0.79 | 61 |

### 3. Hyperparameter Tuning

To improve the model's performance, `GridSearchCV` was used to find the optimal hyperparameters. The following parameters were tuned:
-   `loss`: ['log_loss', 'exponential']
-   `learning_rate`: [0.01, 0.05, 0.1]
-   `n_estimators`:
-   `max_depth`:
-   `subsample`: [0.8, 1.0]

### 4. Tuned Model Evaluation

After fitting the `GridSearchCV` object, the best set of parameters was used to train the final model. This tuned model showed an improvement in overall performance compared to the baseline.

| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Healthy)** | 0.75 | 0.86 | 0.80 | 28 |
| **1 (Heart Disease)** | 0.86 | 0.76 | 0.81 | 33 |
| **accuracy** | | | **0.80** | **61** |
| **macro avg** | 0.81 | 0.81 | 0.80 | 61 |
| **weighted avg** | 0.81 | 0.80 | 0.80 | 61 |

## Conclusion

The exploratory data analysis provided valuable insights into the factors associated with heart disease. Features such as chest pain type, sex, and exercise-induced angina are significant indicators.

The initial Gradient Boosting Classifier provided a solid baseline accuracy of **79%**. The process of hyperparameter tuning with GridSearchCV was a crucial step to further refine the model. After tuning, the model's overall accuracy increased to **80%**. While the accuracy gain is modest, there were notable improvements in other metrics, such as the recall for the "Healthy" class (from 0.82 to 0.86) and the precision for the "Heart Disease" class (from 0.83 to 0.86). This indicates a more balanced and reliable model. This project demonstrates a complete workflow for a classification problem, from data exploration and visualization to model building and optimization.

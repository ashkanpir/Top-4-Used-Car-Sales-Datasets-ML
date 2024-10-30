# Used Car Price Prediction Analysis

This project provides a comprehensive analysis and predictive model training for used car prices using a detailed dataset. Key steps include data exploration, extensive correlation analysis, handling missing data, feature engineering, and model tuning. The primary objective is to identify factors influencing car prices and develop an accurate predictive model.

## Project Overview

In this notebook, we focus on correlation analysis to identify key features impacting car prices. With a series of machine learning models, we assess the predictive power of each model, fine-tuning and optimizing for accuracy. Finally, SHAP values are used to interpret the influence of features on price predictions.

## Data Processing and Analysis

1. **Data Exploration**: Initial inspection of unique counts, missing values, and data structure.
2. **Correlation Analysis**: A detailed correlation study helps prioritize feature engineering and informs model selection.
3. **Data Cleaning**: Imputation and removal of non-informative columns based on missing data and correlation results.
4. **Feature Engineering**: Includes target encoding, one-hot encoding, and selection of high-correlation numeric features.
5. **Modeling**: Training and tuning of Linear Regression, Random Forest, and XGBoost models.

## Models Trained and Evaluated

The following models were trained and evaluated to predict car prices:

- **Linear Regression**: Baseline model performance.
- **Random Forest Regressor**: Captures non-linear relationships with variable importance.
- **XGBoost Regressor**: Optimized for high accuracy and serves as the final model choice.

Each model is evaluated using RMSE and R² scores, with detailed hyperparameter tuning for XGBoost and Random Forest.

## Interpretation with SHAP Values

SHAP values provide feature importance insights for the optimized XGBoost model, offering interpretability for each predictor’s impact on car prices.

## Key Dependencies

- `pandas`, `numpy` for data handling.
- `matplotlib`, `seaborn` for data visualization.
- `scikit-learn` for model training and evaluation.
- `xgboost` for advanced regression modeling.
- `shap` for model interpretability.



```bash
git clone https://github.com/username/repository-name.git

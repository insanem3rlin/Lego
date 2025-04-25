# Price Prediction with Random Forest

This repository contains a Jupyter Notebook that:
1. Fetches lego-set data via the `dbrepo` REST API  
2. Preprocesses and encodes features  
3. Trains a Random Forest regression model  
4. Evaluates performance on validation and test sets  
5. Saves the trained model and a “true vs. predicted” scatter plot  

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Running the Notebook](#running-the-notebook)  
- [Notebook Walkthrough](#notebook-walkthrough)  
  - [1. Imports & Installs](#1-imports--installs)  
  - [2. API Client & Data Loading](#2-api-client--data-loading)  
  - [3. Type Conversion & Cleaning](#3-type-conversion--cleaning)  
  - [4. Categorical Encoding](#4-categorical-encoding)  
  - [5. Train/Test Split & Feature Setup](#5-traintest-split--feature-setup)  
  - [6. Model Pipeline & Training](#6-model-pipeline--training)  
  - [7. Evaluation Metrics](#7-evaluation-metrics)  
  - [8. Artifact Saving](#8-artifact-saving)  
  - [9. Visualization](#9-visualization)  
- [Outputs](#outputs)  
- [Notes & Tips](#notes--tips)  
- [License](#license)  

---

## Prerequisites

- Python 3.13.2  
- Jupyter Notebook or JupyterLab  
- Access to the TU Wien `dbrepo` testing endpoint  

---

## Installation

Installation of packages is done in the first code cell. There also the correct versions are defined. 

> **Tip:** Consider creating a dedicated virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate      # macOS/Linux
> venv\Scripts\activate.bat     # Windows
> ```

---

## Configuration

1. **API Credentials**  
   The notebook instantiates the REST client with:
   ```python
   client = RestClient(
       endpoint="https://test.dbrepo.tuwien.ac.at",
       username="12006729",
       password="Arschbacke11!"
   )
   ```
   For security, you replace hard-coded credentials with environment variables or a secure secrets store:
   ```python
   import os
   client = RestClient(
       endpoint=os.getenv("DBREPO_ENDPOINT"),
       username=os.getenv("DBREPO_USER"),
       password=os.getenv("DBREPO_PASS")
   )
   ```

2. **Identifier IDs**  
   Three UUIDs are used to fetch training, test, and validation splits. Modify if you need other datasets.

---

## Running the Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Lego.ipynb`.  
3. Execute the cells in order.

---

## Notebook Walkthrough

### 1. Imports & Installs
- Installs pinned versions using the `%pip install`.  
- Imports standard libs (`pickle`, `datetime`), HTTP (`requests`), data tools (`pandas`, `numpy`, `matplotlib`), ML (`scikit-learn`), and the custom `RestClient` to acces DBRepo for the 3 datasets.

### 2. API Client & Data Loading
- Instantiates `RestClient`.  
- Uses `get_identifier_data(identifier_id=...)` to pull three DataFrames:  
  - `df_train`  
  - `df_test`  
  - `df_valid`

### 3. Type Conversion & Cleaning
- Ensures numeric columns (`pieces`, `age`, `msrp_int`, `price_int`, `id`) are integers.  
- This prevents downstream dtype issues in modeling.

### 4. Categorical Encoding
- Defines `cat_cols = ['theme','theme_group','subtheme']`.  
- Fits an `OrdinalEncoder` on the training split (`handle_unknown='use_encoded_value'`, unknowns→−1).  
- Transforms all three splits so categories become integer codes.

### 5. Train/Test Split & Feature Setup
- Features: `['theme','theme_group','subtheme','age','pieces','msrp_int']`  
- Targets: `price_int`  
- Assigns `X_train,y_train`, `X_test,y_test`, `X_valid,y_valid`.

### 6. Model Pipeline & Training
- Builds a `Pipeline` with a single step:  
  ```python
  ('rf', RandomForestRegressor(
      n_estimators=200,
      max_depth=10,
      random_state=12006729
  ))
  ```
- Fits the pipeline on `(X_train, y_train)`.

### 7. Evaluation Metrics
- Predicts on **validation** and **test** sets.  
- Computes and prints:
  - Mean Absolute Error (MAE)  
  - Root Mean Squared Error (RMSE)  
  - R² (coefficient of determination)

### 8. Artifact Saving
- **Model** → `rf_price_predictor.pkl` via `pickle.dump`.  
- **Plot** → `val_true_vs_pred.png` saved with `matplotlib`.

### 9. Visualization
- Scatter of true vs. predicted on validation data.  
- Identity line (`y=x`) in red dashed for reference.

---

## Outputs

- **`rf_price_predictor.pkl`** — trained Random Forest pipeline.  
- **`val_true_vs_pred.png`** — scatter-plot of validation predictions.  

You can load the model in another script:
```python
import pickle
with open("rf_price_predictor.pkl","rb") as f:
    model = pickle.load(f)
```

---

## Notes & Tips

- If you encounter any new categorical levels in production data, they’ll map to −1 (the “unknown” bucket). Consider swapping to `OneHotEncoder` or `TargetEncoder` if better performance is needed.  
- To tune hyperparameters, look into `GridSearchCV` or `RandomizedSearchCV` around the `Pipeline`.  

---

## License

This project is licensed under the GPL-3.0 License as seen in the LICENSE file in this repository.  
Feel free to adapt and extend!

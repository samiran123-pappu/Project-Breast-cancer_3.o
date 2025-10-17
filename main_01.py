import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import OrdinalEncoder  # Unc


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline (num_attribs, cat_attribs): 
    # Pipelines
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")), 
        # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
        # 1. Load the data
    housing = pd.read_csv("BRCA.csv")

    # 2. Create a stratified test set based on income category
    housing["age_cat"] = pd.cut(
        housing["Age"],
        bins=[20,30, 40,50,60,70,80,90, np.inf],
        labels=[1, 2, 3, 4, 5,6,7,8]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["age_cat"]):
        housing.loc[test_index].drop("age_cat", axis=1).to_csv("input.csv", index=False)
        housing = housing.loc[train_index].drop("age_cat", axis=1)

# 3. Separate predictors and labels
    housing = housing.dropna(subset=["Patient_Status"])
    housing_labels = housing["Patient_Status"].copy()
    housing_features = housing.drop("Patient_Status", axis=1)

    # ADD THIS LINE: Convert string labels to numerical values
    housing_labels = housing_labels.map({'Alive': 0, 'Dead': 1}) 

    cat_attribs = ["Gender", "Tumour_Stage", "Histology", "ER status", "PR status", "HER2 status", "Surgery_type"]

    # All other columns should be numerical
    num_attribs = [col for col in housing_features.columns if col not in cat_attribs]

    print("Numerical columns:", num_attribs)
    print("Categorical columns:", cat_attribs)


    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = LinearRegression()
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)    
    print("Congrats The model is trained and saved to disk")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["Patient_Status"] = predictions    
    input_data.to_csv("output.csv", index = False)
    print("Inference is done and the results are saved to output.csv")






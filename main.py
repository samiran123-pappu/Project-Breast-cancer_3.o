import pandas as pd
import numpy as np
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
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal

# 1. Load the data
housing = pd.read_csv("BRCA.csv")

# 2. Create a stratified test set based on age category
housing["age_cat"] = pd.cut(
    housing["Age"],
    bins=[20,30, 40,50,60,70,80,90, np.inf],
    labels=[1, 2, 3, 4, 5,6,7,8]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["age_cat"]):
    strat_train_set = housing.loc[train_index].drop("age_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("age_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
# 3. Separate predictors and labels
housing = housing.dropna(subset=["Patient_Status"])
housing_labels = housing["Patient_Status"].copy()
housing_features = housing.drop("Patient_Status", axis=1)

# ADD THIS LINE: Convert string labels to numerical values
housing_labels = housing_labels.map({'Alive': 0, 'Dead': 1})  # Add this line
# print(housing, housing_labels)

# 4. Separate numerical and categorical columns
# Exclude multiple specific columns
cat_attribs = ["Gender", "Tumour_Stage", "Histology", "ER status", "PR status", "HER2 status", "Surgery_type"]

# All other columns should be numerical
num_attribs = [col for col in housing_features.columns if col not in cat_attribs]

print("Numerical columns:", num_attribs)
print("Categorical columns:", cat_attribs)
# print(num_attribs, cat_attribs)

# 5. Pipelines
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

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing_features)

# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)
print(housing_prepared)


# # Convert the prepared NumPy array back to a DataFrame with appropriate column names
# # Get feature names from transformers
# num_features = num_attribs
# cat_features = list(full_pipeline.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_attribs))
# all_features = num_features + cat_features

# housing_prepared_df = pd.DataFrame(housing_prepared, columns=all_features, index=housing.index)
# print(housing_prepared_df.head())

# # housing_prepared_df.to_csv("i.csv", index=False)



# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Linear Regression is: {lin_rmse}")
print("root mean squared error for Linear Regression", pd.Series(lin_rmses).describe())

# Decision Tree Regression
Dec_reg = DecisionTreeRegressor()
Dec_reg.fit(housing_prepared, housing_labels)
Dec_preds = Dec_reg.predict(housing_prepared)
# Dec_rmse = root_mean_squared_error(housing_labels, Dec_preds)
Dec_rmses = -cross_val_score(Dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Decision Tree Regression is: {Dec_rmse}")
print("root mean squared error for Decision Tree Regression", pd.Series(Dec_rmses).describe())


# RandomForestRegressor
Ran_For_reg = RandomForestRegressor()
Ran_For_reg.fit(housing_prepared, housing_labels)
Ran_For_preds = Ran_For_reg.predict(housing_prepared)
# Ran_For_rmse = root_mean_squared_error(housing_labels, Ran_For_preds)
Ran_For_rmses = -cross_val_score(Ran_For_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The root mean squared error for Random Forest Regression is: {Ran_For_rmse}")
print("root mean squared error for Random Forest Regression", pd.Series(Ran_For_rmses).describe())





# Age	Gender	Protein1	Protein2	Protein3	Protein4	Tumour_Stage	Histology	ER status	PR status	HER2 status	Surgery_type	Patient_Status			

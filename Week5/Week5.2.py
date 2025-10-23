# WEEK 5 TASKS: HANDLING MISSING DATA, FEATURE SCALING, AND ENCODING -----------------------------------
# This program covers:
# A. Handling missing data
# B. Replacing missing data with mean/median values
# C. Feature scaling (MinMax, Normalization, Binarization, Standardization)
# D. Encoding categorical data
# ------------------------------------------------------------------------------------------------------

# IMPORT LIBRARIES --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, Normalizer, Binarizer, StandardScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)

# A. HANDLING MISSING DATA -----------------------------------------------------------------------------
# Load the 'pima_indians_diabetes_2.csv' dataset -------------------------------------------------------
pima = pd.read_csv("pima_indians_diabetes_2.csv")

# Check data dimensions and preview --------------------------------------------------------------------
print("=== Data Overview ===")
print("Shape:", pima.shape)
print("\nFirst five rows:\n", pima.head())
print("\nLast five rows:\n", pima.tail())

# Check for missing/null values ------------------------------------------------------------------------
print("\n=== Missing Values (Boolean Mask) ===")
print(pima.isna())

# Display rows that contain missing values -------------------------------------------------------------
print("\n=== Rows with Missing Values ===")
print(pima[pima.isna().any(axis=1)])

# Drop rows with missing values ------------------------------------------------------------------------
pima_dropped = pima.dropna()
print("\n=== After Dropping Rows with Missing Values ===")
print("Shape:", pima_dropped.shape)

# Check again for null values --------------------------------------------------------------------------
print("\nRows with nulls after dropna():")
print(pima_dropped[pima_dropped.isna().any(axis=1)])

# B. MISSING DATA WITH REPLACEMENT ---------------------------------------------------------------------
# Replace missing values using SimpleImputer ------------------------------------------------------------
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_median = SimpleImputer(missing_values=np.nan, strategy='median')

# Apply imputer with mean -------------------------------------------------------------------------------
pima_mean = pd.DataFrame(imputer_mean.fit_transform(pima), columns=pima.columns)
print("\n=== Missing Values Replaced with Mean ===")
print(pima_mean.loc[[440, 661, 770]])

# Apply imputer with median -----------------------------------------------------------------------------
pima_median = pd.DataFrame(imputer_median.fit_transform(pima), columns=pima.columns)
print("\n=== Missing Values Replaced with Median ===")
print(pima_median.loc[[440, 661, 770]])

# C. FEATURE SCALING -----------------------------------------------------------------------------------
# Using Min-Max Scaler ---------------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
pima_scaled = pd.DataFrame(scaler.fit_transform(pima_dropped), columns=pima.columns)
print("\n=== Min-Max Scaled Data ===")
print(pima_scaled.head())

# L1 Normalization -------------------------------------------------------------------------------------
normalizer_l1 = Normalizer(norm='l1')
pima_l1 = pd.DataFrame(normalizer_l1.fit_transform(pima_dropped), columns=pima.columns)
print("\n=== L1 Normalized Data (rows sum to 1) ===")
print(pima_l1.head())

# L2 Normalization -------------------------------------------------------------------------------------
normalizer_l2 = Normalizer(norm='l2')
pima_l2 = pd.DataFrame(normalizer_l2.fit_transform(pima_dropped), columns=pima.columns)
print("\n=== L2 Normalized Data (sum of squares per row = 1) ===")
print(pima_l2.head())

# Binarization -----------------------------------------------------------------------------------------
binarizer = Binarizer(threshold=0.5)
pima_bin = pd.DataFrame(binarizer.fit_transform(pima_scaled), columns=pima.columns)
print("\n=== Binarized Data (Threshold=0.5) ===")
print(pima_bin.tail())

# Standardization --------------------------------------------------------------------------------------
scaler_std = StandardScaler()
pima_standardized = pd.DataFrame(scaler_std.fit_transform(pima_dropped), columns=pima.columns)
print("\n=== Standardized Data (Mean=0, Std=1) ===")
print(pima_standardized.head())

# D. ENCODING CATEGORICAL DATA -------------------------------------------------------------------------
# Load the temperature dataset -------------------------------------------------------------------------
temp = pd.read_csv("temperature.csv")
print("\n=== Temperature Dataset ===")
print(temp.head())

# Label Encoding ---------------------------------------------------------------------------------------
label_encoder = LabelEncoder()
temp["Temperature_LabelEncoded"] = label_encoder.fit_transform(temp["Temperature"])
print("\n=== Label Encoded Temperature ===")
print(temp[["Temperature", "Temperature_LabelEncoded"]])

# Try label encoding with iris dataset -----------------------------------------------------------------
iris = pd.read_csv("iris_data.csv")
label_encoder_iris = LabelEncoder()
iris["iris_class_encoded"] = label_encoder_iris.fit_transform(iris["iris_class"])
print("\n=== Iris Dataset Label Encoded ===")
print(iris[["iris_class", "iris_class_encoded"]].head(3))
print(iris[["iris_class", "iris_class_encoded"]].tail(3))

# Ordinal Encoding -------------------------------------------------------------------------------------
# Define category order for Temperature ----------------------------------------------------------------
ordinal_map = [["Cold", "Warm", "Hot", "Very Hot"]]
ordinal_encoder = OrdinalEncoder(categories=ordinal_map)
temp["Temperature_OrdinalEncoded"] = ordinal_encoder.fit_transform(temp[["Temperature"]])
print("\n=== Ordinal Encoded Temperature (Cold=1, Warm=2, Hot=3, Very Hot=4) ===")
print(temp[["Temperature", "Temperature_OrdinalEncoded"]])

# One-Hot Encoding -------------------------------------------------------------------------------------
onehot_encoder = OneHotEncoder(sparse_output=False)
temp_onehot = pd.DataFrame(onehot_encoder.fit_transform(temp[["Temperature"]]),
                           columns=onehot_encoder.get_feature_names_out(["Temperature"]))
print("\n=== One-Hot Encoded Temperature ===")
print(temp_onehot.head())

# One-Hot Encoding for iris dataset --------------------------------------------------------------------
iris_onehot = pd.DataFrame(onehot_encoder.fit_transform(iris[["iris_class"]]),
                           columns=onehot_encoder.get_feature_names_out(["iris_class"]))
print("\n=== One-Hot Encoded Iris Classes ===")
print(iris_onehot.head())

# END OF WEEK 5 PROGRAM --------------------------------------------------------------------------------
print("\n=== Week 5 Data Preprocessing Complete ===")

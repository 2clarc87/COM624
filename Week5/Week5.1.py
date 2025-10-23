# IMPORT LIBRARIES --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# SIMPLE LINEAR REGRESSION -----------------------------------------------------------------------------
# Provide data ------------------------------------------------------------------------------------------
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))   # X must be 2D
y = np.array([5, 20, 14, 32, 22, 38])

# Create and fit the model -----------------------------------------------------------------------------
model = LinearRegression()
model.fit(x, y)

# Get regression results --------------------------------------------------------------------------------
r_sq = model.score(x, y)
print("=== Linear Regression ===")
print(f"Coefficient of determination (R²): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

# Predict response --------------------------------------------------------------------------------------
y_pred = model.predict(x)
print("Predicted response (Linear):", y_pred, sep="\n")

# Plot results ------------------------------------------------------------------------------------------
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y, color="red", label="Actual")
plt.plot(x, y_pred, color="blue", label="Predicted")
plt.legend()
plt.show()

# POLYNOMIAL REGRESSION ---------------------------------------------------------------------------------
# Provide new data --------------------------------------------------------------------------------------
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# Transform input data to polynomial features (degree 2) ------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Create and fit the model ------------------------------------------------------------------------------
model = LinearRegression().fit(x_poly, y)

# Get regression results --------------------------------------------------------------------------------
r_sq = model.score(x_poly, y)
print("\n=== Polynomial Regression (Degree 2) ===")
print(f"Coefficient of determination (R²): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Predict response --------------------------------------------------------------------------------------
y_pred = model.predict(x_poly)
print("Predicted response (Polynomial):", y_pred, sep="\n")

# Plot results ------------------------------------------------------------------------------------------
plt.title("Polynomial Regression: Actual vs Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y, color="red", label="Actual")
plt.plot(x, y_pred, color="green", label="Predicted")
plt.legend()
plt.show()

# MULTIPLE REGRESSION (with Polynomial Terms) ------------------------------------------------------------
# Provide data (two features) ---------------------------------------------------------------------------
x = np.array([
    [0, 1], [5, 1], [15, 2], [25, 5],
    [35, 11], [45, 15], [55, 34], [60, 35]
])
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])

# Transform input data to polynomial features (degree 2) -------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Create and fit model ----------------------------------------------------------------------------------
model = LinearRegression().fit(x_poly, y)

# Get regression results --------------------------------------------------------------------------------
r_sq = model.score(x_poly, y)
print("\n=== Multiple Polynomial Regression ===")
print(f"Coefficient of determination (R²): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Predict response --------------------------------------------------------------------------------------
y_pred = model.predict(x_poly)
print("Predicted response (Multiple Polynomial):", y_pred, sep="\n")

# CAR CO2 EMISSION EXAMPLE -------------------------------------------------------------------------------
# Load dataset ------------------------------------------------------------------------------------------
df = pd.read_csv("cars.csv")

# Define features (Weight, Volume) and target (CO2) -----------------------------------------------------
x = df[['Weight', 'Volume']]
y = df['CO2']

# Transform features to polynomial features -------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Create and fit model ----------------------------------------------------------------------------------
model = LinearRegression().fit(x_poly, y)

# Get results -------------------------------------------------------------------------------------------
r_sq = model.score(x_poly, y)
print("\n=== Car CO₂ Emission Model ===")
print(f"Coefficient of determination (R²): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Predict CO₂ for all cars in dataset -------------------------------------------------------------------
y_pred = model.predict(x_poly)
print("Predicted CO₂ emissions:", y_pred, sep="\n")

# Predict CO₂ for a specific car ------------------------------------------------------------------------
# Weight = 2300 kg, Volume = 1300 cm³
# Must transform input using the same polynomial transformer
new_data = pd.DataFrame([[2300, 1300]], columns=['Weight', 'Volume'])
new_data_poly = poly.transform(new_data)
predicted_CO2 = model.predict(new_data_poly)
print("\nPredicted CO₂ for car (2300kg, 1300cm³):", predicted_CO2)

# ------------------------------
# STEP 1: Setup - Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")  # For clean plots

# ------------------------------
# STEP 2: Load and Inspect the Dataset
# ------------------------------
df = pd.read_csv('Data/retail_sales_final.csv')
print("First 5 rows:\n", df.head())
print("\nData Summary Info:\n")
df.info()
print("\nSummary Statistics:\n", df.describe(include='all'))

# ------------------------------
# STEP 3: Visualise Missing Data
# ------------------------------
# Bar Chart - Missing per column
df.isnull().sum().plot(kind='bar', color='orange')
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram - Distribution of Sales
df['Sales'].plot(kind='hist', bins=20, color='skyblue')
plt.title("Sales Distribution (Messy)")
plt.xlabel("Sales")
plt.tight_layout()
plt.show()

# Pie chart - Proportion of missing vs non-missing
missing_total = df.isnull().sum().sum()
non_missing_total = df.size - missing_total
plt.pie([missing_total, non_missing_total], labels=['Missing', 'Non-Missing'], autopct='%1.1f%%', colors=['red', 'green'])
plt.title("Overall Missing Data Proportion")
plt.show()

# ------------------------------
# STEP 4: Heatmap & Boxplot (Optional)
# ------------------------------
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

sns.boxplot(x=df['Sales'])
plt.title("Sales Boxplot (Messy)")
plt.show()

# ------------------------------
# STEP 5: Cleaning the Dataset
# ------------------------------
# Remove duplicates
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates()

# Drop rows with missing values
df_cleaned = df.dropna()

# Clean column names
df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')

# ------------------------------
# STEP 6: Central Tendency (Sales)
# ------------------------------
print("Mean Sales:", df_cleaned['sales'].mean())
print("Median Sales:", df_cleaned['sales'].median())
print("Mode Sales:", df_cleaned['sales'].mode()[0])

# ------------------------------
# STEP 7: Full EDA on Cleaned Dataset
# ------------------------------
# Univariate: Histogram
df_cleaned['sales'].plot(kind='hist', bins=20, color='green')
plt.title("Sales Distribution (Clean)")
plt.xlabel("Sales")
plt.tight_layout()
plt.show()

# Univariate: Boxplot
sns.boxplot(x=df_cleaned['sales'])
plt.title("Sales Boxplot (Clean)")
plt.show()

# Multivariate: Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_cleaned[['sales','profit']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Multivariate: Scatterplot
sns.scatterplot(x='sales', y='profit', data=df_cleaned)
plt.title("Sales vs Profit")
plt.show()

# ------------------------------
# STEP 8: Export Cleaned Dataset
# ------------------------------
df_cleaned.to_csv('Data/retail_sales_clean.csv', index=False)
print("Cleaned data exported to Data/retail_sales_clean.csv")

# ------------------------------
# STEP 9: Optional – Fill Missing Values (Examples)
# ------------------------------
# Reload original messy data
df = pd.read_csv('Data/retail_sales_final.csv')

# Example 1: Fill 'Sales' with mean
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())

# Example 2: Fill 'Profit' with 0
df['Profit'] = df['Profit'].fillna(0)

# Example 3: Fill 'Country' with 'Unknown'
df['Country'] = df['Country'].fillna('Unknown')

# Example 4: Forward Fill
df['Sales'] = df['Sales'].ffill()

# Example 5: Backward Fill
df['Sales'] = df['Sales'].bfill()

# Final check for missing values
print("\nMissing values after filling:\n", df.isnull().sum())

# Optional: Save this version too
df.to_csv('Data/retail_sales_filled.csv', index=False)

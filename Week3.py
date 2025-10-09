# ML Week 03: Pandas & Data Cleaning

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. Create your own dataset
# ------------------------------
data = {
    "calories": [420, 380, 390, 230],
    "duration": [50, 40, 45, 20],
    "heartbeat": [120, 110, 115, 90]
}
df = pd.DataFrame(data)
print("Custom Dataset:\n", df)

# Accessing specific data
print("\nFirst row:\n", df.loc[0])
print("\nFirst and last row:\n", df.loc[[0, 3]])
print("\nHeartbeat column:\n", df["heartbeat"])

# ------------------------------
# 2. Save DataFrame to CSV
# ------------------------------
df_indexed = pd.DataFrame(data, index=["day1", "day2", "day3", "day4"])
print("\nData with custom index:\n", df_indexed)

df_indexed.to_csv('Data/with_indexes_and_header.csv', index=True, header=True)
df_indexed.to_csv('Data/no_indexes_with_header.csv', index=False, header=True)
df_indexed.to_csv('Data/no_indexes_no_header.csv', index=False, header=False)

# ------------------------------
# 3. Read CSV and Display
# ------------------------------
df = pd.read_csv('data.csv')
print("\nRead from data.csv:\n", df.head(5))
print(df.tail(10))
print(df.info())

# ------------------------------
# 4. Statistical Summary
# ------------------------------
print("\nStatistical Description:\n", df.describe())

# ------------------------------
# 5. Handle Missing Data
# ------------------------------
# Drop rows with any null values
df = pd.read_csv('data.csv')
print("\nBefore dropping nulls:\n", df.info())
df.dropna(inplace=True)
print("\nAfter dropping nulls:\n", df.info())
df.to_csv('Data/data_no_nulls.csv', index=False)

# Fill missing values with 130
df = pd.read_csv('data.csv')
df.fillna(130, inplace=True)
df.to_csv('Data/data_filled_130.csv', index=False)

# Fill missing Calories with 130 only
df = pd.read_csv('data.csv')
df["Calories"].fillna(130, inplace=True)
df.to_csv('Data/data_filled_calories_130.csv', index=False)

# Fill with MEAN
df = pd.read_csv('data.csv')
mean_val = df["Calories"].mean()
df["Calories"].fillna(mean_val, inplace=True)
df.to_csv('Data/data_filled_mean.csv', index=False)

# Fill with MEDIAN
df = pd.read_csv('data.csv')
median_val = df["Calories"].median()
df["Calories"].fillna(median_val, inplace=True)
df.to_csv('Data/data_filled_median.csv', index=False)

# Fill with MODE
df = pd.read_csv('data.csv')
mode_val = df["Calories"].mode()[0]
df["Calories"].fillna(mode_val, inplace=True)
df.to_csv('Data/data_filled_mode.csv', index=False)

# ------------------------------
# 6. Fixing Data Format
# ------------------------------
df = pd.read_csv('dirtydata.csv')
df['Date'].fillna('2020/12/22', inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('Data/dirtydata_fixed_date.csv', index=False)

# ------------------------------
# 7. Fix Specific Values
# ------------------------------
# Fix row 7 duration to 45
df = pd.read_csv('dirtydata.csv')
df.loc[7, 'Duration'] = 45
df['Date'] = pd.to_datetime(df['Date'])
df.to_csv('Data/dirtydata_duration_fixed.csv', index=False)

# Limit Duration to between 5 and 120
df = pd.read_csv('dirtydata.csv')
for x in df.index:
    if df.loc[x, "Duration"] > 120:
        df.loc[x, "Duration"] = 120
    elif df.loc[x, "Duration"] < 5:
        df.loc[x, "Duration"] = 5
df.to_csv('Data/dirtydata_duration_limited.csv', index=False)

# ------------------------------
# 8. Remove Duplicates
# ------------------------------
df = pd.read_csv('dirtydata.csv')
df.drop_duplicates(inplace=True)
df.to_csv('Data/dirtydata_no_duplicates.csv', index=False)

# ------------------------------
# 9. Merging DataFrames
# ------------------------------
df = pd.read_csv('data.csv')
df1 = df.head()
df2 = df.tail()
df3 = pd.concat([df1, df2], axis=0)
df3.to_csv('Data/data_merged_df3.csv', index=False)

# ------------------------------
# 10. Correlation Matrix
# ------------------------------
df = pd.read_csv('Data/data_filled_mode.csv')
print("\nCorrelation Matrix:\n", df.corr())

# ------------------------------
# 11. Plotting
# ------------------------------
# Line Plot
df.plot()
plt.title("Line Plot")
plt.show()

# Scatter Plot - Duration vs Calories
df.plot(kind='scatter', x='Duration', y='Calories')
plt.title("Scatter: Duration vs Calories")
plt.show()

# Scatter Plot - Duration vs Maxpulse
df.plot(kind='scatter', x='Duration', y='Maxpulse')
plt.title("Scatter: Duration vs Maxpulse")
plt.show()

# Histogram - Duration
df["Duration"].plot(kind='hist')
plt.title("Histogram: Duration")
plt.show()

# Histogram - Calories
df["Calories"].plot(kind='hist')
plt.title("Histogram: Calories")
plt.show()

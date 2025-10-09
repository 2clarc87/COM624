import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/user_behavior_dataset.csv")
df.dropna(inplace=True)
df = df[df['Screen_On_Time'] > 0]

mean_screen = df['Screen_On_Time'].mean()
median_screen = df['Screen_On_Time'].median()
mode_screen = df['Screen_On_Time'].mode()[0]
std_screen = df['Screen_On_Time'].std()
range_screen = df['Screen_On_Time'].max() - df['Screen_On_Time'].min()

print("📊 Descriptive Statistics for Screen-On Time")
print(f"Mean: {mean_screen:.2f} hours/day")
print(f"Median: {median_screen:.2f}")
print(f"Mode: {mode_screen:.2f}")
print(f"Standard Deviation: {std_screen:.2f}")
print(f"Range: {range_screen:.2f}\n")

plt.figure(figsize=(8,5))
sns.histplot(df['Screen_On_Time'], bins=20, kde=True, color='teal')
plt.title("Distribution of Screen-On Time")
plt.xlabel("Hours per Day")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['App_Usage_Time'], color='orange')
plt.title("Boxplot of App Usage Time")
plt.xlabel("Minutes per Day")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df[['App_Usage_Time','Screen_On_Time','Battery_Drain','Number_of_Apps_Installed','Data_Usage','Age','User_Behavior_Class']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.tight_layout()
plt.show()

os_group = df.groupby('Operating_System')['Screen_On_Time'].mean()
os_group.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Average Screen-On Time by Operating System")
plt.ylabel("Hours per Day")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("📋 Summary Table of Dataset")
print(df.describe())
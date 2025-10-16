import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

df = pd.read_csv('social_ads.csv')
print(df.head())
print(df.info())

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))

# Decision boundary
X_set, y_set = X_test, y_test
X1, X2 = X_set[:, 0], X_set[:, 1]
X1_grid, X2_grid = np.meshgrid(
np.arange(start=X1.min()-1, stop=X1.max()+1, step=0.01),
np.arange(start=X2.min()-1, stop=X2.max()+1, step=0.01))


plt.contourf(X1_grid, X2_grid,
model.predict(np.array([X1_grid.ravel(), X2_grid.ravel()]).T).reshape(X1_grid.shape),alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.scatter(X1, X2, c=y_set, edgecolors='k', cmap=ListedColormap(('red', 'green')))
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age (scaled)")
plt.ylabel("Estimated Salary (scaled)")
plt.show()
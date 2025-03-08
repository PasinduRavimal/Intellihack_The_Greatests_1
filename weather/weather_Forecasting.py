
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix  


df = pd.read_csv('weather_data.csv')
print("Data loaded successfully!")

df['avg_temperature'].fillna(df['avg_temperature'].mean(), inplace=True)
df['humidity'].fillna(df['humidity'].mean(), inplace=True)
df['avg_wind_speed'].fillna(df['avg_wind_speed'].mean(), inplace=True)
df['cloud_cover'].fillna(df['cloud_cover'].mean(), inplace=True)
df['pressure'].fillna(df['pressure'].mean(), inplace=True)

print("Missing values handled.")


df['rain_or_not'] = df['rain_or_not'].apply(lambda x: 1 if x == 'Rain' else 0)
print("Rain status converted to numerical values.")


df.dropna(subset=['rain_or_not'], inplace=True)
print("Removed data with missing 'rain_or_not' information.")


df['date'] = pd.to_datetime(df['date'])
print("Date format standardized.")


df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
print("Checked and cleaned illogical entries like negative humidity.")


df.drop(columns=['date'], inplace=True)
print("Irrelevant 'date' column removed.")

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("Correlation matrix generated to understand feature dependencies.")


plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.boxplot(x='rain_or_not', y='avg_temperature', data=df)
plt.subplot(2, 3, 2)
sns.boxplot(x='rain_or_not', y='humidity', data=df)
plt.subplot(2, 3, 3)
sns.boxplot(x='rain_or_not', y='avg_wind_speed', data=df)
plt.subplot(2, 3, 4)
sns.boxplot(x='rain_or_not', y='cloud_cover', data=df)
plt.subplot(2, 3, 5)
sns.boxplot(x='rain_or_not', y='pressure', data=df)
plt.show()
print("Boxplots created to understand feature distributions against rain status.")


X = df.drop(columns=['rain_or_not'])
y = df['rain_or_not']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  
print("Features scaled using StandardScaler.")


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("---")

# --- 4.3.2 Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("---")

# --- 4.3.3 Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("---")


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
print("---")

print("Initial model training and evaluation complete.")


param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10]  #
}


grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("Optimized Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))
print("---")
print("Hyperparameter tuning complete.")


y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
print("Probability of Rain (Random Forest):", y_prob_rf)


conf_matrix = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print("Confusion matrix generated to assess the model's performance.")
print("Project Completed Successfully! ")
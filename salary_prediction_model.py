import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("expected_ctc.csv")

df.drop(columns=['IDX', 'Applicant_ID', 'Inhand_Offer'], errors='ignore', inplace=True)
df.dropna(thresh=int(df.shape[1] * 0.7), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df['Experience_Gap'] = df['Total_Experience'] - df['Total_Experience_in_field_applied']
df['Expected_CTC_Log'] = np.log1p(df['Expected_CTC'])

X = df.drop(columns=['Expected_CTC'])
y = df['Expected_CTC']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n Model Comparison ")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R²:", r2_score(y_test, preds))

best_model = models["Random Forest"]
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print("\nCross-Validation R² Scores (Random Forest):", cv_scores)
print("Mean R²:", np.mean(cv_scores))

param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)

print("Final Tuned Random Forest Model Performance")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

joblib.dump(final_model, 'salary_predictor_model.pkl')
print("\nModel saved as 'salary_predictor_model.pkl'")

sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df['Expected_CTC'], bins=30, kde=True)
plt.title("Expected CTC Distribution")
plt.xlabel("Expected CTC")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
    annot_kws={"size": 6}, linewidths=0.3, cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap", fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Total_Experience'], y=df['Expected_CTC'])
plt.title("Experience vs Expected CTC")
plt.xlabel("Total Experience")
plt.ylabel("Expected CTC")
plt.tight_layout()
plt.show()

importances = final_model.feature_importances_
feat_names = X.columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=feat_names[sorted_idx])
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# ================================
# Chronic Kidney Disease Detection
# ================================

print("CKD Detection script started...")

# Step 1: Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load dataset (auto-detect path)
possible_paths = [
    os.path.join("Dataset", "kidney_disease.csv"),  # inside Dataset folder
    "kidney_disease.csv",                           # same folder as script
    r"C:\Users\Admin\Desktop\CKD_Detection\Dataset\kidney_disease.csv"  # absolute path
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"Dataset found at: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    raise FileNotFoundError("Could not find 'kidney_disease.csv'. Please place it in 'Dataset/' or project root.")

print("Dataset shape:", df.shape)
print(df.head())

# Step 3: Data preprocessing
df_clean = df.copy()

# Replace invalid placeholders with NaN
df_clean.replace(['?', '\t?'], np.nan, inplace=True)

numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')  # convert invalid strings to NaN



# Numeric columns: fill with median
for col in numeric_cols:
    if df_clean[col].notna().any():  # check if there are any valid numbers
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    else:
        df_clean[col].fillna(0, inplace=True)  # if all values are NaN

# Categorical columns: fill with mode
for col in categorical_cols:
    if not df_clean[col].mode().empty:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    else:
        df_clean[col].fillna('unknown', inplace=True)




# Clean extra spaces/tabs in text
for col in ['classification', 'dm', 'cad']:
    df_clean[col] = df_clean[col].astype(str).str.replace('\t', '').str.strip()

# Encode target: notckd=0, ckd=1
df_clean['classification'] = df_clean['classification'].map({'notckd': 0, 'ckd': 1})

# One-hot encoding for categorical features
df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

print("Preprocessing complete. Cleaned dataset shape:", df_clean.shape)

# Step 4: Split data
y = df_clean['classification']

# Drop target + id column (if exists)
X = df_clean.drop(['classification', 'id'], axis=1, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Descriptive Statistics
print("\n=== Dataset Descriptive Statistics ===")
print(df_clean.describe(include='all').T)


# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df["age"].dropna(), kde=True, bins=20, color="skyblue")
plt.title("Age distribution")
plt.xlabel("age")
plt.ylabel("Density")
plt.show()

# Scatter plot: Age vs Blood Pressure
plt.figure(figsize=(5,5))  # plot size
plt.scatter(df['age'], df['bp'], color='blue')  # adjust column names if needed
plt.xlabel('age')            # x-axis label
plt.ylabel('blood pressure') # y-axis label
plt.title('age VS blood Pressure')  # title
plt.show()

# Correlation heatmap (numeric)
plt.figure(figsize=(12,10))
sns.heatmap(df_clean[numeric_cols + ['classification']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation (numeric features + target)")
plt.show()


# Step 5: Train models
# Logistic Regression
lr_model = LogisticRegression(random_state=42, class_weight="balanced")
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Step 6: Evaluation
print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ================================
# Step 6.5: Visualizations
# ================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Confusion Matrices ---
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
plot_confusion_matrix(y_test, rf_pred, "Random Forest")

# --- Metric Comparison (Accuracy & Recall) ---
metrics = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, lr_pred),
        "Recall": recall_score(y_test, lr_pred)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Recall": recall_score(y_test, rf_pred)
    }
}

metrics_df = pd.DataFrame(metrics).T
metrics_df.plot(kind="bar", figsize=(7,5))
plt.title("Model Comparison: Accuracy & Recall")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()

# --- ROC Curves ---
plt.figure(figsize=(7,6))

# Logistic Regression ROC
lr_probs = lr_model.predict_proba(X_test_scaled)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.2f})")

# Random Forest ROC
rf_probs = rf_model.predict_proba(X_test_scaled)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.2f})")

# Random line (baseline)
plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# --- Feature Importance (Random Forest only) ---
importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False).head(15)  # top 15

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Top 15 Important Features - Random Forest")
plt.show()

# Step 7: Save models into Models/ folder
os.makedirs("Models", exist_ok=True)

joblib.dump(rf_model, "Models/kidney_disease_model.pkl")
joblib.dump(scaler, "Models/scaler.pkl")
joblib.dump(X.columns, "Models/feature_columns.pkl")

print("\nModels and preprocessing objects saved in 'Models/' folder")
print("Files created:")
print("- Models/kidney_disease_model.pkl")
print("- Models/scaler.pkl")
print("- Models/feature_columns.pkl")
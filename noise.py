import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('text_features.csv')
X = data.drop(['Movement', 'Author', 'GutenbergID'], axis=1)
y = data['Movement']

# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3037, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Original pipeline
scaler = StandardScaler()
selector = SelectKBest(score_func=f_classif, k=50)
X_train_scaled = scaler.fit_transform(X_train_smote)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_smote)
X_val_scaled = scaler.transform(X_val)
X_val_selected = selector.transform(X_val_scaled)
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# Train original SVM
param_grid = {'C': [0.1, 0.3, 0.5], 'gamma': ['scale', 0.005], 'kernel': ['rbf'], 'class_weight': ['balanced']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_selected, y_train_smote)

# Evaluate original model
best_svm = grid_search.best_estimator_
y_pred_test = best_svm.predict(X_test_selected)
original_accuracy = accuracy_score(y_test, y_pred_test)
original_f1_renaissance = f1_score(y_test, y_pred_test, labels=['Renaissance'], average=None)[0]
print(f"Original SVM - Test Accuracy: {original_accuracy:.4f}, Renaissance F1: {original_f1_renaissance:.4f}")

# Add Gaussian noise (sigma=0.1) to validation and test data
np.random.seed(42)
X_val_noisy = X_val + np.random.normal(0, 0.1, X_val.shape)
X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)

# Scale and select features for noisy data
X_val_noisy_scaled = scaler.transform(X_val_noisy)
X_val_noisy_selected = selector.transform(X_val_noisy_scaled)
X_test_noisy_scaled = scaler.transform(X_test_noisy)
X_test_noisy_selected = selector.transform(X_test_noisy_scaled)

# Train and evaluate with noisy data
grid_search_noisy = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_noisy.fit(X_train_selected, y_train_smote)  # Train on original data
y_pred_test_noisy = grid_search_noisy.best_estimator_.predict(X_test_noisy_selected)
noisy_accuracy = accuracy_score(y_test, y_pred_test_noisy)
noisy_f1_renaissance = f1_score(y_test, y_pred_test_noisy, labels=['Renaissance'], average=None)[0]
print(f"Noisy SVM (sigma=0.1) - Test Accuracy: {noisy_accuracy:.4f}, Renaissance F1: {noisy_f1_renaissance:.4f}")
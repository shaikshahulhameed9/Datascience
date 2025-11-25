"""
Employee Performance & Retention Analysis
Auto-generated script that performs:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Probability & statistical analysis
- Predictive modeling (attrition classification, performance regression)
- Deep Learning examples (Keras)
- Saves key outputs (models, plots) into /mnt/data/output/ by default

Run:
    python3 employee_analysis.py

Requires:
    pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, joblib, scipy

Dataset:
    /mnt/data/employee_data.csv  (change path below if needed)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

# --------------------
DATA_PATH = Path("/mnt/data/employee_data.csv")  # UPDATE if your CSV is elsewhere
OUTPUT_DIR = Path("/mnt/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
def load_data(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def basic_checks(df):
    print("\n=== Basic Info ===")
    print("Shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

def preprocess(df):
    # Drop ID-like columns
    id_cols = [c for c in df.columns if c.lower() in ('id','employeeid','employee_id')]
    df = df.drop(columns=id_cols, errors='ignore')
    # Drop unnamed
    unnamed = [c for c in df.columns if c.lower().startswith('unnamed')]
    df = df.drop(columns=unnamed, errors='ignore')
    # Strip strings
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    # Map Attrition to binary if present
    if 'Attrition' in df.columns:
        df['Attrition_bin'] = df['Attrition'].astype(str).str.lower().map({'yes':1,'no':0})
    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    # Fill categorical NaNs
    cat_cols = df.select_dtypes(include=['object']).columns
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')
    return df

def eda(df):
    print("\n=== EDA ===")
    desc = df.describe(include='all').T
    desc.to_csv(OUTPUT_DIR / "descriptive_stats.csv")
    print("Saved descriptive stats -> output/descriptive_stats.csv")

    # Attrition distribution plot
    if 'Attrition' in df.columns or 'Attrition_bin' in df.columns:
        plt.figure(figsize=(6,4))
        if 'Attrition' in df.columns:
            sns.countplot(x='Attrition', data=df)
            plt.title('Attrition Distribution')
        else:
            sns.countplot(x='Attrition_bin', data=df)
            plt.title('Attrition Distribution (binary)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "attrition_distribution.png")
        plt.close()

    # Correlation heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
        plt.close()

    # Boxplots for first 6 numeric features (outlier detection)
    for col in list(num_cols)[:6]:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"boxplot_{col}.png")
        plt.close()

    print("EDA plots saved to output/")

def probability_and_stats(df):
    print("\n=== Probability & Statistical Analysis ===")
    # P(attrition)
    if 'Attrition' in df.columns:
        p_attr = (df['Attrition'].str.lower()=='yes').mean()
        print("P(Attrition) =", p_attr)
    elif 'Attrition_bin' in df.columns:
        p_attr = df['Attrition_bin'].mean()
        print("P(Attrition) =", p_attr)
    # Bayes example: P(attrition | low performance)
    if 'PerformanceScore' in df.columns:
        median_perf = df['PerformanceScore'].median()
        low_perf_mask = df['PerformanceScore'] < median_perf
        if 'Attrition' in df.columns:
            p_attr_given_low = df[low_perf_mask]['Attrition'].str.lower().eq('yes').mean()
        elif 'Attrition_bin' in df.columns:
            p_attr_given_low = df[low_perf_mask]['Attrition_bin'].mean()
        else:
            p_attr_given_low = None
        print("P(Attrition | Performance < median) =", p_attr_given_low)
    # ANOVA across departments (if available)
    try:
        from scipy.stats import f_oneway
        if 'Department' in df.columns and 'PerformanceScore' in df.columns:
            groups = [g['PerformanceScore'].values for _, g in df.groupby('Department')]
            if len(groups) > 1:
                stat, p = f_oneway(*groups)
                print("ANOVA across departments: F-statistic = {:.4f} p-value = {:.4f}".format(stat, p))
    except Exception as e:
        print("ANOVA skipped (scipy missing or error):", e)

def prepare_features(df, drop_target=None):
    X = df.copy()
    if drop_target and drop_target in X.columns:
        X = X.drop(columns=[drop_target])
    # Label-encode categorical columns simply
    cat_cols = X.select_dtypes(include=['object']).columns
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        le_map[c] = le
    X_num = X.select_dtypes(include=[np.number])
    return X_num, le_map

def attrition_model(df):
    print("\n=== Attrition Prediction (Random Forest) ===")
    if 'Attrition_bin' not in df.columns:
        print("Attrition column not found. Skipping attrition model.")
        return
    X, le_map = prepare_features(df, drop_target='Attrition_bin')
    y = df['Attrition_bin']
    if 'Attrition_bin' in X.columns:
        X = X.drop(columns=['Attrition_bin'], errors=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Attrition)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attrition_confusion_matrix.png")
    plt.close()
    joblib.dump(rf, OUTPUT_DIR / "rf_attrition_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "rf_attrition_scaler.joblib")
    print("Saved Random Forest model and scaler to output/")

def performance_regression(df):
    print("\n=== Performance Regression (Linear Regression) ===")
    if 'PerformanceScore' not in df.columns:
        print("PerformanceScore not found. Skipping regression.")
        return
    X, le_map = prepare_features(df, drop_target='PerformanceScore')
    y = df['PerformanceScore']
    if 'PerformanceScore' in X.columns:
        X = X.drop(columns=['PerformanceScore'], errors=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)
    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual PerformanceScore")
    plt.ylabel("Predicted PerformanceScore")
    plt.title("Predicted vs Actual Performance")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='grey')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_pred_vs_actual.png")
    plt.close()
    joblib.dump(lr, OUTPUT_DIR / "linear_regression_performance.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "linear_regression_scaler.joblib")
    print("Saved Linear Regression model and scaler to output/")

def deep_learning_regression(df, epochs=20):
    print("\n=== Deep Learning Regression (Keras) ===")
    if 'PerformanceScore' not in df.columns:
        print("PerformanceScore not found. Skipping DL regression.")
        return
    X, le_map = prepare_features(df, drop_target='PerformanceScore')
    y = df['PerformanceScore'].values
    if 'PerformanceScore' in X.columns:
        X = X.drop(columns=['PerformanceScore'], errors=False)
    if X.shape[1] == 0:
        print("No numeric features for DL regression.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=epochs, batch_size=16, verbose=1)
    test_mse = model.evaluate(X_test_s, y_test, verbose=0)[0]
    print("DL Test MSE:", test_mse)
    model.save(OUTPUT_DIR / "dl_performance_model.keras")
    joblib.dump(scaler, OUTPUT_DIR / "dl_performance_scaler.joblib")
    print("Saved DL regression model and scaler to output/")

def deep_learning_classification(df, epochs=15):
    print("\n=== Deep Learning Classification (Attrition) ===")
    if 'Attrition_bin' not in df.columns:
        print("Attrition column missing. Skipping DL classification.")
        return
    X, le_map = prepare_features(df, drop_target='Attrition_bin')
    y = df['Attrition_bin'].values
    if 'Attrition_bin' in X.columns:
        X = X.drop(columns=['Attrition_bin'], errors=False)
    if X.shape[1] == 0:
        print("No numeric features for DL classification.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train, validation_split=0.2, epochs=epochs, batch_size=16, verbose=1)
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print("DL Test Accuracy:", acc)
    model.save(OUTPUT_DIR / "dl_attrition_model.keras")
    joblib.dump(scaler, OUTPUT_DIR / "dl_attrition_scaler.joblib")
    print("Saved DL classification model and scaler to output/")

def main():
    df = load_data()
    basic_checks(df)
    df = preprocess(df)
    eda(df)
    probability_and_stats(df)
    attrition_model(df)
    performance_regression(df)
    # quick DL runs (lower epochs) — increase epochs for final training
    try:
        deep_learning_regression(df, epochs=10)
        deep_learning_classification(df, epochs=10)
    except Exception as e:
        print("Deep Learning training skipped or failed:", e)
    print("\nDone. Check the output folder for artifacts.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPRI Hackathon – Exploration & Model-Planning Phase
=====================================================
Comprehensive EDA, feature engineering, mathematical analysis,
feature ranking, baseline modelling, and experiment-matrix generation.

This script generates ALL required outputs under analysis/ and notebooks/.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, hashlib, itertools, textwrap

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
)
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures, SplineTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, ElasticNet, HuberRegressor, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    IsolationForest
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import permutation_importance

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

print("="*80)
print("CPRI Hackathon – Exploration & Model-Planning Phase")
print("="*80)
print(f"Started: {datetime.now()}")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
XLSX = Path(r"d:\Powernext\dataset\CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx")
OUT  = Path(r"d:\Powernext\analysis")
CHARTS = OUT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
(Path(r"d:\Powernext\notebooks")).mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.max_open_warning": 0, "savefig.dpi": 150, "savefig.bbox": "tight"})

SENSORS   = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
OPERATING = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
NUMERIC   = OPERATING + SENSORS
TARGET_CLS = "Validity_Label"
TARGET_REG = "Reference_Parameter"

# collectors for reports
findings = {}        # data-quality findings
interpretations = {} # chart interpretations
relationship_notes = []  # mathematical analysis notes

# ─────────────────────────────────────────────
# 0. Load data
# ─────────────────────────────────────────────
print("\n[0] Loading data …")
xls = pd.ExcelFile(XLSX)
print(f"  Sheet names: {xls.sheet_names}")

train = pd.read_excel(xls, "Training_Data")
test  = pd.read_excel(xls, "Test_Data")
sample_sub = pd.read_excel(xls, "Sample_Submission")
readme_df  = pd.read_excel(xls, "README")

print(f"  Training shape : {train.shape}")
print(f"  Test shape     : {test.shape}")
print(f"  Sample sub     : {sample_sub.shape}")
print(f"  README         : {readme_df.shape}")

findings["train_shape"] = str(train.shape)
findings["test_shape"]  = str(test.shape)

# ╔══════════════════════════════════════════╗
# ║  SECTION 1 – DATA-QUALITY AUDIT         ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 1 – DATA-QUALITY AUDIT")
print("="*80)

# 1a. Column names and dtypes
print("\n[1a] Columns & dtypes")
print(train.dtypes.to_string())
findings["train_columns"] = list(train.columns)
findings["test_columns"]  = list(test.columns)
findings["train_dtypes"]  = {c: str(d) for c, d in train.dtypes.items()}

# 1b. Missing values
print("\n[1b] Missing values")
miss_train = train.isnull().sum()
miss_test  = test.isnull().sum()
miss_pct_train = (train.isnull().mean()*100).round(2)
print("Training missing counts:\n", miss_train[miss_train > 0].to_string())
print("Training missing %:\n", miss_pct_train[miss_pct_train > 0].to_string())
print("Test missing counts:\n", miss_test[miss_test > 0].to_string())

findings["train_missing"] = miss_train.to_dict()
findings["test_missing"]  = miss_test.to_dict()

# 1c. Missingness by Validity_Label
print("\n[1c] Missingness by Validity_Label")
for lbl in train[TARGET_CLS].unique():
    sub = train[train[TARGET_CLS] == lbl]
    m = sub.isnull().sum()
    m = m[m > 0]
    if len(m):
        print(f"  {lbl} ({len(sub)} rows): {m.to_dict()}")

miss_by_label = {}
for lbl in train[TARGET_CLS].unique():
    sub = train[train[TARGET_CLS] == lbl]
    miss_by_label[lbl] = sub.isnull().sum().to_dict()
findings["missingness_by_label"] = miss_by_label

# 1d. Unique values
print("\n[1d] Unique values per column")
for c in train.columns:
    print(f"  {c}: {train[c].nunique()}")

# 1e. Class balance
vc = train[TARGET_CLS].value_counts()
print(f"\n[1e] Class balance:\n{vc.to_string()}")
findings["class_balance"] = vc.to_dict()

# 1f. Exact duplicates (full row)
full_dups = train.duplicated(keep=False)
print(f"\n[1f] Exact full-row duplicates: {full_dups.sum()}")
findings["exact_full_row_dups"] = int(full_dups.sum())

# 1g. Duplicate Test_IDs
dup_ids = train["Test_ID"].duplicated(keep=False)
print(f"[1g] Duplicate Test_IDs: {dup_ids.sum()}")
findings["dup_test_ids"] = int(dup_ids.sum())

# 1h. Duplicate measurement vectors (excl ID & targets)
meas_cols = OPERATING + SENSORS
dup_meas = train.duplicated(subset=meas_cols, keep=False)
n_dup_meas = dup_meas.sum()
print(f"[1h] Duplicate measurement vectors (excl ID & targets): {n_dup_meas}")
findings["dup_measurement_vectors"] = int(n_dup_meas)

if n_dup_meas > 0:
    dup_groups = train[dup_meas].groupby(meas_cols, dropna=False)
    dup_group_sizes = dup_groups.size().reset_index(name="count")
    n_pairs = len(dup_group_sizes)
    print(f"  Number of unique duplicate groups: {n_pairs}")
    findings["dup_measurement_groups"] = int(n_pairs)

    # Check conflicting targets
    dup_rows = train[dup_meas].copy()
    dup_rows["_grp"] = dup_rows.groupby(meas_cols, dropna=False).ngroup()
    conflict_ref = dup_rows.groupby("_grp")[TARGET_REG].nunique()
    conflict_lbl = dup_rows.groupby("_grp")[TARGET_CLS].nunique()
    n_conflict_ref = (conflict_ref > 1).sum()
    n_conflict_lbl = (conflict_lbl > 1).sum()
    print(f"  Groups with conflicting Reference_Parameter: {n_conflict_ref}")
    print(f"  Groups with conflicting Validity_Label: {n_conflict_lbl}")
    findings["dup_conflicting_ref_param"] = int(n_conflict_ref)
    findings["dup_conflicting_validity"]  = int(n_conflict_lbl)

    # Labels of duplicates
    dup_labels = dup_rows[TARGET_CLS].value_counts()
    print(f"  Validity of duplicate rows: {dup_labels.to_dict()}")
    findings["dup_labels"] = dup_labels.to_dict()

# Test data duplicates
dup_meas_test = test.duplicated(subset=[c for c in meas_cols if c in test.columns], keep=False)
print(f"  Test duplicate measurement vectors: {dup_meas_test.sum()}")
findings["test_dup_measurement_vectors"] = int(dup_meas_test.sum())

# 1i. Near-duplicates at different rounding tolerances
print("\n[1i] Near-duplicates at rounding tolerances")
for decimals in [3, 2, 1, 0]:
    rounded = train[meas_cols].round(decimals)
    nd = rounded.duplicated(keep=False).sum()
    print(f"  {decimals} decimal places: {nd} rows in near-duplicate groups")
findings["near_dup_by_rounding"] = {}
for decimals in [3, 2, 1, 0]:
    rounded = train[meas_cols].round(decimals)
    nd = rounded.duplicated(keep=False).sum()
    findings["near_dup_by_rounding"][str(decimals)] = int(nd)

# 1j. Negative and zero sensor values
print("\n[1j] Negative and zero sensor values")
for s in SENSORS:
    neg = (train[s] < 0).sum()
    zero = (train[s] == 0).sum()
    if neg > 0 or zero > 0:
        print(f"  {s}: negative={neg}, zero={zero}")
findings["negative_sensor_values"] = {s: int((train[s] < 0).sum()) for s in SENSORS}
findings["zero_sensor_values"]     = {s: int((train[s] == 0).sum()) for s in SENSORS}

# Also for operating columns
for c in OPERATING:
    neg = (train[c] < 0).sum()
    zero = (train[c] == 0).sum()
    if neg > 0 or zero > 0:
        print(f"  {c}: negative={neg}, zero={zero}")

# 1k. Clipping check (min/max piling)
print("\n[1k] Value clipping check (min/max piling)")
for c in NUMERIC:
    vals = train[c].dropna()
    if len(vals) == 0:
        continue
    min_count = (vals == vals.min()).sum()
    max_count = (vals == vals.max()).sum()
    if min_count > 5 or max_count > 5:
        print(f"  {c}: min={vals.min():.4f} (count={min_count}), max={vals.max():.4f} (count={max_count})")

# 1l. Extreme values – IQR, robust z-score, MAD
print("\n[1l] Extreme value analysis")
extreme_summary = {}
for c in NUMERIC:
    vals = train[c].dropna()
    if len(vals) == 0:
        continue
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = ((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum()

    med = vals.median()
    mad = np.median(np.abs(vals - med))
    mad_scale = mad * 1.4826 if mad > 0 else vals.std()
    robust_z = np.abs((vals - med) / mad_scale) if mad_scale > 0 else pd.Series([0]*len(vals))
    robust_z_outliers = (robust_z > 3).sum()

    extreme_summary[c] = {
        "iqr_outliers": int(iqr_outliers),
        "robust_z_outliers": int(robust_z_outliers),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(med),
        "std": float(vals.std()),
    }
    if iqr_outliers > 0 or robust_z_outliers > 0:
        print(f"  {c}: IQR outliers={iqr_outliers}, Robust-Z outliers={robust_z_outliers}")

findings["extreme_values"] = extreme_summary

# 1m. Train vs Test distributions
print("\n[1m] Train vs Test distribution comparison (KS test)")
ks_results = {}
common_cols = [c for c in NUMERIC if c in test.columns]
for c in common_cols:
    t1 = train[c].dropna()
    t2 = test[c].dropna()
    if len(t1) > 0 and len(t2) > 0:
        stat, pval = stats.ks_2samp(t1, t2)
        ks_results[c] = {"statistic": round(stat, 4), "p_value": round(pval, 4)}
        if pval < 0.05:
            print(f"  {c}: KS stat={stat:.4f}, p={pval:.4f} *** SIGNIFICANT DRIFT ***")
        else:
            print(f"  {c}: KS stat={stat:.4f}, p={pval:.4f}")
findings["ks_train_test"] = ks_results

# Summary stats
desc_train = train[NUMERIC].describe().T
desc_train.to_csv(OUT / "train_descriptive_stats.csv")

# ╔══════════════════════════════════════════╗
# ║  SECTION 2 – EXPLORATORY CHARTS         ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 2 – EXPLORATORY CHARTS")
print("="*80)

# Chart 1: Class balance
fig, ax = plt.subplots(figsize=(6, 4))
vc.plot(kind="bar", color=["#2ecc71", "#e74c3c"], ax=ax)
ax.set_title("Valid vs Invalid Class Counts")
ax.set_ylabel("Count")
for i, v in enumerate(vc.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
fig.savefig(CHARTS / "01_class_balance.png")
plt.close(fig)
interpretations["01_class_balance"] = (
    f"Class imbalance: {vc.iloc[0]} Valid vs {vc.iloc[1]} Invalid "
    f"({vc.iloc[1]/len(train)*100:.1f}% minority). "
    "Moderate imbalance – class-weighted models and PR-AUC evaluation are advisable."
)
print(f"  [1] Class balance: {interpretations['01_class_balance']}")

# Chart 2: Missing values
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
miss_train_nz = miss_train[miss_train > 0]
if len(miss_train_nz) > 0:
    miss_train_nz.plot(kind="bar", ax=axes[0], color="#3498db")
    axes[0].set_title("Missing Value Counts (Training)")
    axes[0].set_ylabel("Count")
    miss_pct_nz = miss_pct_train[miss_pct_train > 0]
    miss_pct_nz.plot(kind="bar", ax=axes[1], color="#e67e22")
    axes[1].set_title("Missing Value % (Training)")
    axes[1].set_ylabel("Percentage")
else:
    axes[0].text(0.5, 0.5, "No missing values", ha="center", va="center", transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, "No missing values", ha="center", va="center", transform=axes[1].transAxes)
fig.savefig(CHARTS / "02_missing_values.png")
plt.close(fig)
interpretations["02_missing_values"] = (
    f"Missing data concentrated in sensor columns. "
    f"Sensor_S4 has the most missingness ({miss_train.get('Sensor_S4', 0)} values). "
    "Operating conditions have no missing values."
)
print(f"  [2] Missing values: {interpretations['02_missing_values']}")

# Chart 3: Missingness vs Validity
fig, ax = plt.subplots(figsize=(8, 4))
miss_by_label_df = pd.DataFrame(miss_by_label).T
miss_by_label_df = miss_by_label_df[[c for c in miss_by_label_df.columns if miss_by_label_df[c].sum() > 0]]
if len(miss_by_label_df.columns) > 0:
    miss_by_label_df.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Validity Label")
    ax.set_ylabel("Missing Count")
fig.savefig(CHARTS / "03_missingness_vs_validity.png")
plt.close(fig)
interpretations["03_missingness_vs_validity"] = (
    "Missingness may be more prevalent in Invalid records, suggesting it is an informative signal. "
    "Missing-indicator features should be created."
)
print(f"  [3] Missingness vs Validity: {interpretations['03_missingness_vs_validity']}")

# Chart 4: Histograms & KDE for every numeric feature
print("  [4] Histograms & KDE …")
for c in NUMERIC + [TARGET_REG]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train[c].dropna().hist(bins=40, ax=axes[0], color="#3498db", edgecolor="white", alpha=0.7)
    axes[0].set_title(f"{c} – Histogram")
    try:
        train[c].dropna().plot.kde(ax=axes[1], color="#e74c3c")
        axes[1].set_title(f"{c} – KDE")
    except Exception:
        axes[1].set_title(f"{c} – KDE (insufficient data)")
    fig.savefig(CHARTS / f"04_hist_kde_{c}.png")
    plt.close(fig)

# Chart 5: Train vs Test overlay
print("  [5] Train vs Test overlays …")
for c in common_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    train[c].dropna().hist(bins=40, ax=ax, alpha=0.5, label="Train", color="#3498db", density=True)
    test[c].dropna().hist(bins=40, ax=ax, alpha=0.5, label="Test", color="#e74c3c", density=True)
    ax.set_title(f"{c} – Train vs Test Distribution")
    ax.legend()
    fig.savefig(CHARTS / f"05_train_test_{c}.png")
    plt.close(fig)
interpretations["05_train_test_overlay"] = (
    "Train and test distributions appear similar for most features. "
    "KS tests above highlight any statistically significant drifts. "
    "If drift is minimal, standard cross-validation is reliable."
)

# Chart 6: Boxplots & Violin plots by Validity
print("  [6] Boxplots & Violin plots …")
for c in NUMERIC + [TARGET_REG]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=train, x=TARGET_CLS, y=c, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title(f"{c} – Boxplot by Validity")
    sns.violinplot(data=train, x=TARGET_CLS, y=c, ax=axes[1], palette=["#2ecc71", "#e74c3c"], inner="quartile")
    axes[1].set_title(f"{c} – Violin by Validity")
    fig.savefig(CHARTS / f"06_box_violin_{c}.png")
    plt.close(fig)

# Chart 7: Pearson correlation heatmap
print("  [7] Pearson correlation …")
corr_pearson = train[NUMERIC + [TARGET_REG]].corr(method="pearson")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Pearson Correlation Heatmap (Training)")
fig.savefig(CHARTS / "07_pearson_corr.png")
plt.close(fig)

# Chart 8: Spearman correlation
print("  [8] Spearman correlation …")
corr_spearman = train[NUMERIC + [TARGET_REG]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Spearman Correlation Heatmap (Training)")
fig.savefig(CHARTS / "08_spearman_corr.png")
plt.close(fig)

# Chart 9: Correlation – Valid only
print("  [9] Correlation – Valid only …")
valid_data = train[train[TARGET_CLS] == "Valid"]
corr_valid = valid_data[NUMERIC + [TARGET_REG]].corr(method="pearson")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_valid, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Pearson Correlation – Valid Records Only")
fig.savefig(CHARTS / "09_corr_valid_only.png")
plt.close(fig)

# Chart 10: Correlation – Invalid only
print("  [10] Correlation – Invalid only …")
invalid_data = train[train[TARGET_CLS] == "Invalid"]
corr_invalid = invalid_data[NUMERIC + [TARGET_REG]].corr(method="pearson")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_invalid, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Pearson Correlation – Invalid Records Only")
fig.savefig(CHARTS / "10_corr_invalid_only.png")
plt.close(fig)
interpretations["corr_valid_vs_invalid"] = (
    "Comparing Valid-only and Invalid-only correlations reveals whether Invalid records "
    "break the normal operating relationships. Divergent patterns indicate that sensor "
    "agreement and physical consistency differ between the two classes."
)

# Chart 11: Every raw feature vs Reference_Parameter
print("  [11] Feature vs Reference_Parameter …")
for c in NUMERIC:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train[c], train[TARGET_REG], alpha=0.4, s=15, c="#3498db")
    ax.set_xlabel(c)
    ax.set_ylabel(TARGET_REG)
    ax.set_title(f"{c} vs {TARGET_REG}")
    fig.savefig(CHARTS / f"11_vs_refparam_{c}.png")
    plt.close(fig)

# Chart 12: Scatter with Validity colour
print("  [12] Scatter coloured by Validity …")
colour_map = {"Valid": "#2ecc71", "Invalid": "#e74c3c"}
for c in NUMERIC:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, colour in colour_map.items():
        sub = train[train[TARGET_CLS] == lbl]
        ax.scatter(sub[c], sub[TARGET_REG], alpha=0.4, s=15, c=colour, label=lbl)
    ax.set_xlabel(c)
    ax.set_ylabel(TARGET_REG)
    ax.set_title(f"{c} vs {TARGET_REG} (coloured by Validity)")
    ax.legend()
    fig.savefig(CHARTS / f"12_scatter_validity_{c}.png")
    plt.close(fig)

# Chart 13: Polynomial trend curves
print("  [13] Polynomial trend curves …")
for c in NUMERIC:
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = train[[c, TARGET_REG]].dropna().index
    x = train.loc[mask, c].values
    y = train.loc[mask, TARGET_REG].values
    ax.scatter(x, y, alpha=0.3, s=10, c="#bdc3c7")
    if len(x) > 10:
        sort_idx = np.argsort(x)
        for deg, color in [(1, "#3498db"), (2, "#e74c3c"), (3, "#2ecc71")]:
            try:
                coeffs = np.polyfit(x[sort_idx], y[sort_idx], deg)
                yhat = np.polyval(coeffs, x[sort_idx])
                ax.plot(x[sort_idx], yhat, color=color, label=f"Degree {deg}", linewidth=2)
            except Exception:
                pass
    ax.set_xlabel(c)
    ax.set_ylabel(TARGET_REG)
    ax.set_title(f"Polynomial trends: {c} vs {TARGET_REG}")
    ax.legend()
    fig.savefig(CHARTS / f"13_polytrend_{c}.png")
    plt.close(fig)

# Chart 14: Sensor pair plots
print("  [14] Sensor pair plots …")
sensor_data = train[SENSORS + [TARGET_CLS]].dropna()
if len(sensor_data) > 20:
    g = sns.pairplot(sensor_data, hue=TARGET_CLS, palette=colour_map,
                     plot_kws={"alpha": 0.4, "s": 12}, diag_kind="kde", height=2.5)
    g.savefig(CHARTS / "14_sensor_pairplot.png")
    plt.close()
interpretations["14_sensor_pairplot"] = (
    "Sensor pair plots reveal how sensors covary. Invalid records that deviate from "
    "the main sensor-covariance cloud indicate faulty measurements."
)

# Chart 15: Pairwise sensor difference distributions
print("  [15] Sensor difference distributions …")
sensor_pairs = list(itertools.combinations(SENSORS, 2))
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for idx, (s1, s2) in enumerate(sensor_pairs):
    diff = train[s1] - train[s2]
    if idx < len(axes):
        for lbl, colour in colour_map.items():
            sub_diff = diff[train[TARGET_CLS] == lbl].dropna()
            if len(sub_diff) > 2:
                axes[idx].hist(sub_diff, bins=30, alpha=0.5, color=colour, label=lbl, density=True)
        axes[idx].set_title(f"{s1} - {s2}")
        axes[idx].legend(fontsize=7)
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Pairwise Sensor Differences by Validity", y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "15_sensor_diffs.png")
plt.close(fig)

# Chart 16: Voltage-Current operating map
print("  [16] Voltage-Current operating map …")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(train["Applied_Voltage_kV"], train["Load_Current_A"], alpha=0.5, s=20, c="#3498db")
ax.set_xlabel("Applied Voltage (kV)")
ax.set_ylabel("Load Current (A)")
ax.set_title("Voltage–Current Operating Map")
fig.savefig(CHARTS / "16_V_I_map.png")
plt.close(fig)

# Chart 17: Current-Duration operating map
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(train["Load_Current_A"], train["Test_Duration_min"], alpha=0.5, s=20, c="#3498db")
ax.set_xlabel("Load Current (A)")
ax.set_ylabel("Test Duration (min)")
ax.set_title("Current–Duration Operating Map")
fig.savefig(CHARTS / "17_I_t_map.png")
plt.close(fig)

# Chart 18: Operating maps coloured by Ref Param
print("  [18] Operating maps by Reference Parameter …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc1 = axes[0].scatter(train["Applied_Voltage_kV"], train["Load_Current_A"],
                       c=train[TARGET_REG], cmap="viridis", alpha=0.5, s=20)
axes[0].set_xlabel("Voltage (kV)"); axes[0].set_ylabel("Current (A)")
axes[0].set_title("V-I Map coloured by Ref Param")
plt.colorbar(sc1, ax=axes[0])
sc2 = axes[1].scatter(train["Load_Current_A"], train["Test_Duration_min"],
                       c=train[TARGET_REG], cmap="viridis", alpha=0.5, s=20)
axes[1].set_xlabel("Current (A)"); axes[1].set_ylabel("Duration (min)")
axes[1].set_title("I-t Map coloured by Ref Param")
plt.colorbar(sc2, ax=axes[1])
fig.savefig(CHARTS / "18_opmaps_refparam.png")
plt.close(fig)

# Chart 19: Operating maps coloured by validity
print("  [19] Operating maps by Validity …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for lbl, colour in colour_map.items():
    sub = train[train[TARGET_CLS] == lbl]
    axes[0].scatter(sub["Applied_Voltage_kV"], sub["Load_Current_A"],
                    alpha=0.4, s=15, c=colour, label=lbl)
    axes[1].scatter(sub["Load_Current_A"], sub["Test_Duration_min"],
                    alpha=0.4, s=15, c=colour, label=lbl)
axes[0].set_title("V-I Map by Validity"); axes[0].legend()
axes[1].set_title("I-t Map by Validity"); axes[1].legend()
fig.savefig(CHARTS / "19_opmaps_validity.png")
plt.close(fig)

# Chart 20: Duplicate investigation
print("  [20] Duplicate investigation …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train["_is_dup"] = dup_meas.astype(int)
for is_d, colour, lbl in [(1, "#e74c3c", "Duplicate"), (0, "#3498db", "Unique")]:
    sub = train[train["_is_dup"] == is_d]
    axes[0].scatter(sub["Applied_Voltage_kV"], sub["Load_Current_A"],
                    alpha=0.5, s=20, c=colour, label=lbl)
    axes[1].scatter(sub[TARGET_REG], sub["Sensor_S1"],
                    alpha=0.5, s=20, c=colour, label=lbl)
axes[0].set_title("V-I Map: Duplicates highlighted"); axes[0].legend()
axes[1].set_title("Ref Param vs S1: Duplicates highlighted"); axes[1].legend()
fig.savefig(CHARTS / "20_duplicates.png")
plt.close(fig)
train.drop(columns=["_is_dup"], inplace=True)

# Chart 21: Train/test drift plots
print("  [21] Train/test drift …")
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.ravel()
for idx, c in enumerate(common_cols[:8]):
    train[c].dropna().plot.kde(ax=axes[idx], label="Train", color="#3498db")
    test[c].dropna().plot.kde(ax=axes[idx], label="Test", color="#e74c3c")
    axes[idx].set_title(c)
    axes[idx].legend(fontsize=7)
fig.suptitle("Train vs Test KDE Overlays", y=1.02)
fig.tight_layout()
fig.savefig(CHARTS / "21_drift.png")
plt.close(fig)

# Chart 22: Missingness pattern heatmap
print("  [22] Missingness pattern heatmap …")
fig, ax = plt.subplots(figsize=(12, 6))
miss_matrix = train[NUMERIC + [TARGET_REG]].isnull().astype(int)
sns.heatmap(miss_matrix.T, cbar=False, cmap="YlOrRd", ax=ax, yticklabels=True)
ax.set_title("Missingness Pattern Heatmap (Training)")
ax.set_xlabel("Sample index")
fig.savefig(CHARTS / "22_missingness_heatmap.png")
plt.close(fig)
interpretations["22_missingness_heatmap"] = (
    "The missingness heatmap reveals whether missing values co-occur across sensors. "
    "Patterns of joint missingness may indicate systematic measurement failures."
)

print("  Charts saved to", CHARTS)

# ╔══════════════════════════════════════════╗
# ║  SECTION 3 – MATHEMATICAL RELATIONSHIPS ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 3 – MATHEMATICAL RELATIONSHIP ANALYSIS")
print("="*80)

# --- 3a. Regression relationships ---
print("\n[3a] Regression feature analysis")

reg_analysis = {}
for c in NUMERIC:
    mask = train[[c, TARGET_REG]].dropna().index
    x = train.loc[mask, c].values
    y = train.loc[mask, TARGET_REG].values
    if len(x) < 10:
        continue

    # Pearson
    r_p, p_p = stats.pearsonr(x, y)
    # Spearman
    r_s, p_s = stats.spearmanr(x, y)

    # Univariate linear regression R²
    X_lr = x.reshape(-1, 1)
    lr = LinearRegression().fit(X_lr, y)
    r2_lr = r2_score(y, lr.predict(X_lr))

    # Polynomial R² (deg 2)
    poly2 = np.polyfit(x, y, 2)
    r2_poly2 = r2_score(y, np.polyval(poly2, x))

    reg_analysis[c] = {
        "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 6),
        "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 6),
        "linear_R2": round(r2_lr, 4),
        "poly2_R2": round(r2_poly2, 4),
        "n": len(x)
    }
    print(f"  {c:30s}: Pearson={r_p:.3f} Spearman={r_s:.3f} Lin-R²={r2_lr:.3f} Poly2-R²={r2_poly2:.3f}")

# Mutual information for regression
print("\n  Mutual information (regression) …")
mi_data = train[NUMERIC + [TARGET_REG]].dropna()
if len(mi_data) > 50:
    mi_reg = mutual_info_regression(mi_data[NUMERIC], mi_data[TARGET_REG], random_state=42)
    for c, mi_val in zip(NUMERIC, mi_reg):
        reg_analysis[c]["mutual_info"] = round(mi_val, 4)
        print(f"    {c:30s}: MI = {mi_val:.4f}")

# Partial correlations (controlling for operating conditions)
print("\n  Partial correlations (sensor → Ref Param | operating) …")
from numpy.linalg import lstsq
for s in SENSORS:
    mask = train[OPERATING + [s, TARGET_REG]].dropna().index
    if len(mask) < 30:
        continue
    X_ctrl = train.loc[mask, OPERATING].values
    y_s = train.loc[mask, s].values
    y_r = train.loc[mask, TARGET_REG].values
    # Residualize sensor
    beta_s = lstsq(np.column_stack([X_ctrl, np.ones(len(X_ctrl))]), y_s, rcond=None)[0]
    resid_s = y_s - np.column_stack([X_ctrl, np.ones(len(X_ctrl))]) @ beta_s
    # Residualize target
    beta_r = lstsq(np.column_stack([X_ctrl, np.ones(len(X_ctrl))]), y_r, rcond=None)[0]
    resid_r = y_r - np.column_stack([X_ctrl, np.ones(len(X_ctrl))]) @ beta_r
    pcorr, _ = stats.pearsonr(resid_s, resid_r)
    reg_analysis[s]["partial_corr_ctrl_ops"] = round(pcorr, 4)
    print(f"    {s:30s}: partial r = {pcorr:.4f}")

# Bootstrap CIs for top correlations
print("\n  Bootstrap 95% CIs for Pearson correlations …")
np.random.seed(42)
n_boot = 1000
for c in NUMERIC:
    mask = train[[c, TARGET_REG]].dropna().index
    x = train.loc[mask, c].values
    y = train.loc[mask, TARGET_REG].values
    if len(x) < 30:
        continue
    boot_r = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        r, _ = stats.pearsonr(x[idx], y[idx])
        boot_r.append(r)
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    reg_analysis[c]["pearson_ci_lo"] = round(ci_lo, 4)
    reg_analysis[c]["pearson_ci_hi"] = round(ci_hi, 4)
    print(f"    {c:30s}: [{ci_lo:.4f}, {ci_hi:.4f}]")

# Cross-validation stability
print("\n  Correlation stability across 5 folds …")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for c in NUMERIC:
    mask = train[[c, TARGET_REG, TARGET_CLS]].dropna().index
    x = train.loc[mask, c].values
    y = train.loc[mask, TARGET_REG].values
    labels = train.loc[mask, TARGET_CLS].values
    fold_rs = []
    for tr_idx, te_idx in kf.split(x, labels):
        if len(x[te_idx]) > 5:
            r, _ = stats.pearsonr(x[te_idx], y[te_idx])
            fold_rs.append(r)
    if fold_rs:
        reg_analysis[c]["cv_pearson_mean"] = round(np.mean(fold_rs), 4)
        reg_analysis[c]["cv_pearson_std"]  = round(np.std(fold_rs), 4)

# VIF
print("\n  Variance Inflation Factors …")
vif_data = train[NUMERIC].dropna()
if len(vif_data) > 50:
    vif_df = pd.DataFrame()
    vif_vals = []
    for i, c in enumerate(NUMERIC):
        try:
            v = variance_inflation_factor(vif_data.values, i)
            vif_vals.append({"feature": c, "VIF": round(v, 2)})
        except Exception:
            vif_vals.append({"feature": c, "VIF": np.nan})
    vif_df = pd.DataFrame(vif_vals)
    print(vif_df.to_string(index=False))
    findings["VIF"] = vif_vals

# --- 3b. Classification relationships ---
print("\n[3b] Classification feature analysis")

cls_analysis = {}
train["_label_bin"] = (train[TARGET_CLS] == "Invalid").astype(int)

for c in NUMERIC:
    mask = train[[c, "_label_bin"]].dropna().index
    x = train.loc[mask, c].values
    y = train.loc[mask, "_label_bin"].values
    if len(x) < 10:
        continue

    # Point-biserial correlation
    rpb, ppb = stats.pointbiserialr(y, x)

    # Cohen's d
    g1 = x[y == 0]
    g2 = x[y == 1]
    pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
    cohens_d = (g2.mean() - g1.mean()) / pooled_std if pooled_std > 0 else 0

    # Univariate ROC-AUC
    try:
        auc_val = roc_auc_score(y, x)
        auc_val = max(auc_val, 1 - auc_val)  # direction-agnostic
    except Exception:
        auc_val = 0.5

    cls_analysis[c] = {
        "point_biserial_r": round(rpb, 4),
        "point_biserial_p": round(ppb, 6),
        "cohens_d": round(cohens_d, 4),
        "univariate_auc": round(auc_val, 4),
        "n": len(x)
    }
    print(f"  {c:30s}: r_pb={rpb:.3f} d={cohens_d:.3f} AUC={auc_val:.3f}")

# MI for classification
print("\n  Mutual information (classification) …")
mi_cls_data = train[NUMERIC + ["_label_bin"]].dropna()
if len(mi_cls_data) > 50:
    mi_cls = mutual_info_classif(mi_cls_data[NUMERIC], mi_cls_data["_label_bin"], random_state=42)
    for c, mi_val in zip(NUMERIC, mi_cls):
        cls_analysis[c]["mutual_info"] = round(mi_val, 4)

# Invalid rate by feature bins
print("\n  Invalid rate by feature quartiles …")
for c in NUMERIC:
    mask = train[[c, "_label_bin"]].dropna().index
    sub = train.loc[mask, [c, "_label_bin"]].copy()
    try:
        sub["_bin"] = pd.qcut(sub[c], 4, duplicates="drop")
        rates = sub.groupby("_bin")["_label_bin"].mean()
        cls_analysis[c]["invalid_rate_by_quartile"] = {str(k): round(v, 4) for k, v in rates.items()}
    except Exception:
        pass

# Missingness association with Invalid
print("\n  Missingness association with Invalid …")
for c in SENSORS:
    is_miss = train[c].isnull().astype(int)
    if is_miss.sum() > 0 and is_miss.sum() < len(train):
        ct = pd.crosstab(is_miss, train["_label_bin"])
        chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        cls_analysis[c]["miss_chi2"] = round(chi2, 4)
        cls_analysis[c]["miss_chi2_p"] = round(p_chi, 6)
        print(f"  {c} missingness vs Invalid: χ²={chi2:.3f}, p={p_chi:.4f}")

# Duplicate association with Invalid
print("\n  Duplicate association with Invalid …")
is_dup = dup_meas.astype(int)
ct_dup = pd.crosstab(is_dup, train["_label_bin"])
if ct_dup.shape == (2, 2):
    chi2_dup, p_dup, _, _ = stats.chi2_contingency(ct_dup)
    print(f"  Duplicate vs Invalid: χ²={chi2_dup:.3f}, p={p_dup:.4f}")
    findings["dup_invalid_chi2"] = round(chi2_dup, 4)
    findings["dup_invalid_p"]    = round(p_dup, 6)

reg_analysis_df = pd.DataFrame(reg_analysis).T
cls_analysis_df = pd.DataFrame(cls_analysis).T

train.drop(columns=["_label_bin"], inplace=True)

# ╔══════════════════════════════════════════╗
# ║  SECTION 4 – FEATURE ENGINEERING        ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 4 – FEATURE ENGINEERING")
print("="*80)

def engineer_features(df):
    """Create engineered features for a dataframe."""
    f = pd.DataFrame(index=df.index)

    V = df["Applied_Voltage_kV"]
    I = df["Load_Current_A"]
    T = df["Ambient_Temperature_C"]
    t = df["Test_Duration_min"]

    # Electrical / operating features
    f["Power_VI"]       = V * I
    f["Energy_VIt"]     = V * I * t
    f["I_squared"]      = I ** 2
    f["I2t"]            = I ** 2 * t
    f["V_squared"]      = V ** 2
    f["sqrt_t"]         = np.sqrt(t.clip(lower=0))
    f["log1p_t"]        = np.log1p(t.clip(lower=0))
    f["V_over_I"]       = V / I.replace(0, np.nan)
    f["I_over_V"]       = I / V.replace(0, np.nan)
    f["V_I_interact"]   = V * I
    f["V_t_interact"]   = V * t
    f["I_t_interact"]   = I * t
    f["T_I_interact"]   = T * I
    f["T_t_interact"]   = T * t

    # Sensor aggregation features
    sensor_df = df[SENSORS]
    f["sensor_mean"]    = sensor_df.mean(axis=1)
    f["sensor_median"]  = sensor_df.median(axis=1)
    f["sensor_min"]     = sensor_df.min(axis=1)
    f["sensor_max"]     = sensor_df.max(axis=1)
    f["sensor_range"]   = f["sensor_max"] - f["sensor_min"]
    f["sensor_std"]     = sensor_df.std(axis=1)
    f["sensor_var"]     = sensor_df.var(axis=1)
    f["sensor_mad"]     = sensor_df.apply(lambda row: np.nanmedian(np.abs(row - np.nanmedian(row))), axis=1)
    f["sensor_cv"]      = f["sensor_std"] / f["sensor_mean"].replace(0, np.nan)
    # Trimmed mean (exclude min and max)
    def trimmed_mean(row):
        vals = row.dropna().sort_values()
        if len(vals) > 2:
            return vals.iloc[1:-1].mean()
        return vals.mean()
    f["sensor_trimmed_mean"] = sensor_df.apply(trimmed_mean, axis=1)
    f["n_sensors_available"] = sensor_df.notna().sum(axis=1)
    f["n_sensors_missing"]   = sensor_df.isna().sum(axis=1)

    # Sensor missingness indicators
    for s in SENSORS:
        f[f"miss_{s}"] = df[s].isna().astype(int)

    # Negative/zero sensor indicators
    for s in SENSORS:
        f[f"neg_{s}"]  = (df[s] < 0).astype(int)
        f[f"zero_{s}"] = (df[s] == 0).astype(int)

    # Pairwise sensor differences & ratios
    for s1, s2 in itertools.combinations(SENSORS, 2):
        f[f"diff_{s1}_{s2}"]     = df[s1] - df[s2]
        f[f"absdiff_{s1}_{s2}"]  = (df[s1] - df[s2]).abs()
        denom = df[s2].replace(0, np.nan)
        f[f"ratio_{s1}_{s2}"]    = df[s1] / denom

    # Sensor minus ambient
    for s in SENSORS:
        f[f"{s}_minus_ambient"] = df[s] - T

    # Sensor deviation from median
    for s in SENSORS:
        f[f"{s}_dev_from_median"] = df[s] - f["sensor_median"]

    f["max_dev_from_median"] = pd.concat(
        [f[f"{s}_dev_from_median"].abs() for s in SENSORS], axis=1
    ).max(axis=1)

    # Number of sensors outside robust consensus (>2 MAD from median)
    def sensors_outside_consensus(row):
        vals = row.dropna()
        if len(vals) < 2:
            return 0
        med = vals.median()
        mad = np.median(np.abs(vals - med))
        if mad == 0:
            mad = vals.std() * 0.6745
        if mad == 0:
            return 0
        return ((np.abs(vals - med) / mad) > 2).sum()
    f["n_sensors_outside_consensus"] = sensor_df.apply(sensors_outside_consensus, axis=1)

    # Duplicate flags
    meas_cols_list = OPERATING + SENSORS
    avail_meas = [c for c in meas_cols_list if c in df.columns]
    f["is_exact_dup"] = df.duplicated(subset=avail_meas, keep=False).astype(int)
    # Group size
    grp = df.groupby(avail_meas, dropna=False).transform("size") if len(avail_meas) > 0 else 1
    if isinstance(grp, pd.DataFrame):
        grp = grp.iloc[:, 0]
    f["dup_group_size"] = grp

    # Missingness pattern (hash)
    miss_pattern = df[SENSORS].isna().astype(int).apply(lambda r: "".join(map(str, r)), axis=1)
    f["miss_pattern_code"] = miss_pattern.astype("category").cat.codes

    return f

print("  Engineering features …")
train_feat = engineer_features(train)
test_feat  = engineer_features(test)
print(f"  Engineered feature count: {train_feat.shape[1]}")

# Cross-fitted sensor residuals (leave-one-out style via KFold)
print("\n  Cross-fitted sensor residuals …")
kf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for target_sensor in SENSORS:
    other_sensors = [s for s in SENSORS if s != target_sensor]
    predictor_cols = OPERATING + other_sensors
    mask = train[predictor_cols + [target_sensor]].dropna().index
    if len(mask) < 50:
        train_feat[f"residual_{target_sensor}"] = np.nan
        continue
    residuals = pd.Series(np.nan, index=train.index)
    X_all = train.loc[mask, predictor_cols].values
    y_all = train.loc[mask, target_sensor].values
    labels = train.loc[mask, TARGET_CLS].values
    for tr_idx, te_idx in kf5.split(X_all, labels):
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(X_all[tr_idx], y_all[tr_idx])
        pred = rf.predict(X_all[te_idx])
        residuals.iloc[mask[te_idx]] = y_all[te_idx] - pred
    train_feat[f"residual_{target_sensor}"] = residuals
    # Also compute abs residual
    train_feat[f"abs_residual_{target_sensor}"] = residuals.abs()
    print(f"    residual_{target_sensor}: computed for {mask.shape[0]} rows")

# For test data, fit on full training and predict
for target_sensor in SENSORS:
    other_sensors = [s for s in SENSORS if s != target_sensor]
    predictor_cols = OPERATING + other_sensors
    mask_tr = train[predictor_cols + [target_sensor]].dropna().index
    mask_te = test[predictor_cols + [target_sensor]].dropna().index
    if len(mask_tr) < 50 or len(mask_te) < 5:
        test_feat[f"residual_{target_sensor}"] = np.nan
        test_feat[f"abs_residual_{target_sensor}"] = np.nan
        continue
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(train.loc[mask_tr, predictor_cols].values, train.loc[mask_tr, target_sensor].values)
    pred = rf.predict(test.loc[mask_te, predictor_cols].values)
    residuals_te = pd.Series(np.nan, index=test.index)
    residuals_te.iloc[mask_te] = test.loc[mask_te, target_sensor].values - pred
    test_feat[f"residual_{target_sensor}"] = residuals_te
    test_feat[f"abs_residual_{target_sensor}"] = residuals_te.abs()

# Robust outlier scores (Isolation Forest on operating+sensors for Valid training data)
print("\n  Robust outlier scores …")
valid_mask = train[TARGET_CLS] == "Valid"
iso_cols = OPERATING + SENSORS
iso_data_valid = train.loc[valid_mask, iso_cols].dropna()
if len(iso_data_valid) > 50:
    scaler = StandardScaler()
    X_valid_scaled = scaler.fit_transform(iso_data_valid)
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X_valid_scaled)

    # Score all training rows that have complete data
    complete_mask = train[iso_cols].dropna().index
    X_all_scaled = scaler.transform(train.loc[complete_mask, iso_cols])
    iso_scores = iso.decision_function(X_all_scaled)
    train_feat.loc[complete_mask, "isolation_score"] = iso_scores

    # Score test
    complete_mask_te = test[iso_cols].dropna().index
    if len(complete_mask_te) > 0:
        X_te_scaled = scaler.transform(test.loc[complete_mask_te, iso_cols])
        test_feat.loc[complete_mask_te, "isolation_score"] = iso.decision_function(X_te_scaled)

# Build feature catalogue
print("\n  Building feature catalogue …")
feature_catalog = []
all_eng_features = list(train_feat.columns)
for f_name in all_eng_features:
    cat_entry = {
        "feature_name": f_name,
        "n_missing_train": int(train_feat[f_name].isna().sum()),
        "n_missing_test": int(test_feat[f_name].isna().sum()) if f_name in test_feat.columns else "N/A",
    }
    vals = train_feat[f_name].dropna()
    if len(vals) > 0 and vals.dtype in [np.float64, np.int64, np.float32, np.int32]:
        cat_entry["mean"]  = round(float(vals.mean()), 4)
        cat_entry["std"]   = round(float(vals.std()), 4)
        cat_entry["min"]   = round(float(vals.min()), 4)
        cat_entry["max"]   = round(float(vals.max()), 4)
    feature_catalog.append(cat_entry)

feature_catalog_df = pd.DataFrame(feature_catalog)
feature_catalog_df.to_csv(OUT / "feature_catalog.csv", index=False)
print(f"  Feature catalogue saved: {len(feature_catalog_df)} features")

# ╔══════════════════════════════════════════╗
# ║  SECTION 5 – GENUINE REGIMES vs ERRORS  ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 5 – GENUINE REGIMES vs SENSOR ERRORS")
print("="*80)

valid_only = train[train[TARGET_CLS] == "Valid"]
invalid_only = train[train[TARGET_CLS] == "Invalid"]

# Compare anomaly detectors on Valid data
print("\n  Anomaly detection on Valid records …")
anom_cols = OPERATING + SENSORS
valid_complete = valid_only[anom_cols].dropna()
if len(valid_complete) > 50:
    scaler_v = StandardScaler()
    X_vc = scaler_v.fit_transform(valid_complete)

    # Isolation Forest
    iso_v = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_labels = iso_v.fit_predict(X_vc)
    print(f"    Isolation Forest: {(iso_labels == -1).sum()} anomalies in {len(X_vc)} Valid records")

    # LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_labels = lof.fit_predict(X_vc)
    print(f"    LOF: {(lof_labels == -1).sum()} anomalies in {len(X_vc)} Valid records")

    # Robust covariance
    try:
        rc = EllipticEnvelope(contamination=0.05, random_state=42)
        rc_labels = rc.fit_predict(X_vc)
        print(f"    Robust Covariance: {(rc_labels == -1).sum()} anomalies in {len(X_vc)} Valid records")
    except Exception as e:
        print(f"    Robust Covariance failed: {e}")

    # One-Class SVM (on smaller subset for speed)
    if len(X_vc) > 200:
        svm_sample = X_vc[:200]
    else:
        svm_sample = X_vc
    try:
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        oc_svm.fit(svm_sample)
        svm_labels = oc_svm.predict(X_vc)
        print(f"    One-Class SVM: {(svm_labels == -1).sum()} anomalies in {len(X_vc)} Valid records")
    except Exception as e:
        print(f"    One-Class SVM failed: {e}")

# Characterize Invalid vs Valid patterns
print("\n  Invalid record characteristics …")
for c in NUMERIC:
    v_vals = valid_only[c].dropna()
    i_vals = invalid_only[c].dropna()
    if len(v_vals) > 5 and len(i_vals) > 5:
        t_stat, t_p = stats.ttest_ind(v_vals, i_vals, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(v_vals, i_vals, alternative="two-sided")
        print(f"    {c:30s}: Welch t={t_stat:+.3f} (p={t_p:.4f}), Mann-Whitney p={u_p:.4f}")

# ╔══════════════════════════════════════════╗
# ║  SECTION 6 – FEATURE RANKING            ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 6 – FEATURE RANKING")
print("="*80)

# Combine raw + engineered features for ranking
ALL_RAW = NUMERIC.copy()
train_all = pd.concat([train[NUMERIC], train_feat], axis=1)
test_all  = pd.concat([test[[c for c in NUMERIC if c in test.columns]], test_feat], axis=1)

# Remove duplicate columns
train_all = train_all.loc[:, ~train_all.columns.duplicated()]
test_all  = test_all.loc[:, ~test_all.columns.duplicated()]

all_feature_names = list(train_all.columns)
print(f"  Total features for ranking: {len(all_feature_names)}")

# --- 6a. Regression ranking ---
print("\n[6a] Regression feature ranking")

# Prepare data: complete cases for regression
y_reg = train[TARGET_REG]
reg_mask = y_reg.notna()
X_reg = train_all.loc[reg_mask].copy()
y_reg = y_reg.loc[reg_mask].copy()

# Fill NaN for tree models (using median)
X_reg_filled = X_reg.fillna(X_reg.median())

# Remove constant columns
const_cols = [c for c in X_reg_filled.columns if X_reg_filled[c].nunique() <= 1]
if const_cols:
    X_reg_filled.drop(columns=const_cols, inplace=True)
    print(f"  Dropped {len(const_cols)} constant columns")

reg_feature_names = list(X_reg_filled.columns)

# Correlation with target
reg_corrs = X_reg_filled.corrwith(y_reg).abs().sort_values(ascending=False)

# Mutual information
print("  MI for regression …")
mi_reg_all = mutual_info_regression(X_reg_filled, y_reg, random_state=42, n_neighbors=5)
mi_reg_series = pd.Series(mi_reg_all, index=reg_feature_names).sort_values(ascending=False)

# Random Forest importance
print("  RF importance for regression …")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_reg_filled, y_reg)
rf_imp_reg = pd.Series(rf_reg.feature_importances_, index=reg_feature_names).sort_values(ascending=False)

# Extra Trees importance
print("  ET importance for regression …")
et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
et_reg.fit(X_reg_filled, y_reg)
et_imp_reg = pd.Series(et_reg.feature_importances_, index=reg_feature_names).sort_values(ascending=False)

# Gradient Boosting importance
print("  GB importance for regression …")
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_reg.fit(X_reg_filled, y_reg)
gb_imp_reg = pd.Series(gb_reg.feature_importances_, index=reg_feature_names).sort_values(ascending=False)

# Permutation importance (using RF)
print("  Permutation importance for regression …")
perm_reg = permutation_importance(rf_reg, X_reg_filled, y_reg, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp_reg = pd.Series(perm_reg.importances_mean, index=reg_feature_names).sort_values(ascending=False)

# SHAP (lightweight, using small GB model)
print("  SHAP values for regression …")
try:
    import shap
    shap_model_reg = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    shap_model_reg.fit(X_reg_filled, y_reg)
    explainer_reg = shap.TreeExplainer(shap_model_reg)
    shap_values_reg = explainer_reg.shap_values(X_reg_filled.iloc[:200])
    shap_imp_reg = pd.Series(np.abs(shap_values_reg).mean(axis=0), index=reg_feature_names).sort_values(ascending=False)
except Exception as e:
    print(f"  SHAP failed: {e}")
    shap_imp_reg = pd.Series(dtype=float)

# Compile regression ranking
reg_ranking = pd.DataFrame({
    "abs_correlation": reg_corrs,
    "mutual_info": mi_reg_series,
    "rf_importance": rf_imp_reg,
    "et_importance": et_imp_reg,
    "gb_importance": gb_imp_reg,
    "perm_importance": perm_imp_reg,
})
if len(shap_imp_reg) > 0:
    reg_ranking["shap_importance"] = shap_imp_reg

# Rank each method
for col in reg_ranking.columns:
    reg_ranking[f"{col}_rank"] = reg_ranking[col].rank(ascending=False)

# Average rank
rank_cols = [c for c in reg_ranking.columns if c.endswith("_rank")]
reg_ranking["avg_rank"] = reg_ranking[rank_cols].mean(axis=1)
reg_ranking = reg_ranking.sort_values("avg_rank")
reg_ranking.to_csv(OUT / "regression_feature_ranking.csv")
print(f"\n  Top 20 regression features:")
print(reg_ranking.head(20)[["abs_correlation", "mutual_info", "rf_importance", "avg_rank"]].to_string())

# --- 6b. Classification ranking ---
print("\n[6b] Classification feature ranking")

y_cls = (train[TARGET_CLS] == "Invalid").astype(int)
X_cls = train_all.copy()
X_cls_filled = X_cls.fillna(X_cls.median())

# Remove constant columns
const_cols_cls = [c for c in X_cls_filled.columns if X_cls_filled[c].nunique() <= 1]
if const_cols_cls:
    X_cls_filled.drop(columns=const_cols_cls, inplace=True)

cls_feature_names = list(X_cls_filled.columns)

# Point-biserial correlation
cls_corrs = pd.Series({c: abs(stats.pointbiserialr(y_cls, X_cls_filled[c].values)[0])
                        for c in cls_feature_names if X_cls_filled[c].std() > 0},
                       name="abs_pointbiserial")

# MI
print("  MI for classification …")
mi_cls_all = mutual_info_classif(X_cls_filled, y_cls, random_state=42, n_neighbors=5)
mi_cls_series = pd.Series(mi_cls_all, index=cls_feature_names)

# RF
print("  RF importance for classification …")
rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
rf_cls.fit(X_cls_filled, y_cls)
rf_imp_cls = pd.Series(rf_cls.feature_importances_, index=cls_feature_names)

# ET
print("  ET importance for classification …")
et_cls = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42,
                               class_weight="balanced", n_jobs=-1)
et_cls.fit(X_cls_filled, y_cls)
et_imp_cls = pd.Series(et_cls.feature_importances_, index=cls_feature_names)

# GB
print("  GB importance for classification …")
gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_cls.fit(X_cls_filled, y_cls)
gb_imp_cls = pd.Series(gb_cls.feature_importances_, index=cls_feature_names)

# Permutation importance
print("  Permutation importance for classification …")
perm_cls = permutation_importance(rf_cls, X_cls_filled, y_cls, n_repeats=10,
                                   random_state=42, scoring="f1", n_jobs=-1)
perm_imp_cls = pd.Series(perm_cls.importances_mean, index=cls_feature_names)

# SHAP
print("  SHAP values for classification …")
try:
    shap_model_cls = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    shap_model_cls.fit(X_cls_filled, y_cls)
    explainer_cls = shap.TreeExplainer(shap_model_cls)
    shap_values_cls = explainer_cls.shap_values(X_cls_filled.iloc[:200])
    if isinstance(shap_values_cls, list):
        shap_values_cls = shap_values_cls[1]
    shap_imp_cls = pd.Series(np.abs(shap_values_cls).mean(axis=0), index=cls_feature_names)
except Exception as e:
    print(f"  SHAP failed: {e}")
    shap_imp_cls = pd.Series(dtype=float)

# Compile classification ranking
cls_ranking = pd.DataFrame({
    "abs_pointbiserial": cls_corrs,
    "mutual_info": mi_cls_series,
    "rf_importance": rf_imp_cls,
    "et_importance": et_imp_cls,
    "gb_importance": gb_imp_cls,
    "perm_importance": perm_imp_cls,
})
if len(shap_imp_cls) > 0:
    cls_ranking["shap_importance"] = shap_imp_cls

for col in cls_ranking.columns:
    cls_ranking[f"{col}_rank"] = cls_ranking[col].rank(ascending=False)

rank_cols_cls = [c for c in cls_ranking.columns if c.endswith("_rank")]
cls_ranking["avg_rank"] = cls_ranking[rank_cols_cls].mean(axis=1)
cls_ranking = cls_ranking.sort_values("avg_rank")
cls_ranking.to_csv(OUT / "classification_feature_ranking.csv")
print(f"\n  Top 20 classification features:")
print(cls_ranking.head(20)[["abs_pointbiserial", "mutual_info", "rf_importance", "avg_rank"]].to_string())

# ╔══════════════════════════════════════════╗
# ║  SECTION 7 – LIGHTWEIGHT BASELINES      ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 7 – LIGHTWEIGHT BASELINE EXPERIMENTS")
print("="*80)

# Create duplicate-aware groups for CV
print("  Creating duplicate-aware CV groups …")
meas_key = train[meas_cols].fillna(-9999).apply(lambda r: hash(tuple(r)), axis=1)
group_map = {}
grp_id = 0
seen = {}
groups = []
for idx, key in meas_key.items():
    if key in seen:
        groups.append(seen[key])
    else:
        seen[key] = grp_id
        groups.append(grp_id)
        grp_id += 1
train["_cv_group"] = groups

# Feature sets
operating_cols    = OPERATING
sensor_cols       = SENSORS
all_raw_cols      = NUMERIC
electrical_feats  = ["Power_VI", "Energy_VIt", "I_squared", "I2t", "V_squared",
                     "sqrt_t", "log1p_t", "V_over_I", "I_over_V",
                     "V_I_interact", "V_t_interact", "I_t_interact", "T_I_interact", "T_t_interact"]
sensor_agg_feats  = ["sensor_mean", "sensor_median", "sensor_min", "sensor_max",
                     "sensor_range", "sensor_std", "sensor_var", "sensor_mad", "sensor_cv",
                     "sensor_trimmed_mean", "n_sensors_available", "n_sensors_missing"]
sensor_consist    = [c for c in train_feat.columns if c.startswith(("diff_", "absdiff_", "ratio_",
                     "residual_", "abs_residual_", "Sensor_S") ) and "dev_from_median" in c or
                     c in ["max_dev_from_median", "n_sensors_outside_consensus"]]
sensor_consist    = [c for c in train_feat.columns if any(c.startswith(p) for p in
                     ["diff_", "absdiff_", "ratio_", "residual_", "abs_residual_"]) or
                     c in ["max_dev_from_median", "n_sensors_outside_consensus"] or
                     "_dev_from_median" in c or "_minus_ambient" in c]

# Top-ranked compact set (top 15 from rankings)
top_reg_feats = list(reg_ranking.head(15).index)
top_cls_feats = list(cls_ranking.head(15).index)

# All engineered
all_eng_cols = list(train_feat.columns)

# All excluding S4
no_s4_raw = [c for c in NUMERIC if c != "Sensor_S4"]
no_s4_eng = [c for c in all_eng_cols if "S4" not in c and "Sensor_S4" not in c]

def get_feature_set(name):
    """Return feature column names for a named set."""
    sets = {
        "raw_operating": OPERATING,
        "raw_sensors": SENSORS,
        "all_raw": NUMERIC,
        "ops_plus_electrical": OPERATING + electrical_feats,
        "sensors_plus_consistency": SENSORS + sensor_agg_feats + sensor_consist,
        "all_engineered": all_eng_cols,
        "no_S4": no_s4_raw + no_s4_eng,
        "top_compact_reg": top_reg_feats,
        "top_compact_cls": top_cls_feats,
    }
    return sets.get(name, [])

# Helper: prepare X from combined frame
def prepare_X(feature_list, source_raw=train, source_eng=train_feat):
    raw_cols = [c for c in feature_list if c in source_raw.columns and c not in ["Test_ID", TARGET_CLS, TARGET_REG]]
    eng_cols = [c for c in feature_list if c in source_eng.columns]
    parts = []
    if raw_cols:
        parts.append(source_raw[raw_cols])
    if eng_cols:
        parts.append(source_eng[eng_cols])
    if not parts:
        return pd.DataFrame()
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X

# --- Regression baselines ---
print("\n  Regression baselines …")
reg_feature_sets = {
    "raw_operating": OPERATING,
    "raw_sensors": SENSORS,
    "all_raw": NUMERIC,
    "ops_plus_electrical": OPERATING + electrical_feats,
    "sensors_plus_consistency": SENSORS + sensor_agg_feats + sensor_consist,
    "all_engineered": NUMERIC + all_eng_cols,
    "no_S4": no_s4_raw + no_s4_eng,
    "top_compact": top_reg_feats,
}

reg_baseline_results = []
gkf = GroupKFold(n_splits=5)

for fs_name, fs_cols in reg_feature_sets.items():
    X = prepare_X(fs_cols)
    mask = X.notna().all(axis=1) & y_reg.notna() if X.shape[1] < 20 else pd.Series(True, index=X.index)
    X_use = X.fillna(X.median())
    y_use = train[TARGET_REG]
    valid_idx = X_use.index[y_use.notna()]
    X_use = X_use.loc[valid_idx]
    y_use = y_use.loc[valid_idx]
    groups_use = train.loc[valid_idx, "_cv_group"]

    if len(X_use) < 50 or X_use.shape[1] == 0:
        continue

    # HistGradientBoosting (handles NaN natively)
    hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
    try:
        mae_scores = []
        r2_scores = []
        for tr_idx, te_idx in gkf.split(X_use, groups=groups_use):
            hgb.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
            pred = hgb.predict(X_use.iloc[te_idx])
            mae_scores.append(mean_absolute_error(y_use.iloc[te_idx], pred))
            r2_scores.append(r2_score(y_use.iloc[te_idx], pred))
        reg_baseline_results.append({
            "feature_set": fs_name,
            "model": "HistGBR",
            "n_features": X_use.shape[1],
            "MAE_mean": round(np.mean(mae_scores), 4),
            "MAE_std": round(np.std(mae_scores), 4),
            "R2_mean": round(np.mean(r2_scores), 4),
            "R2_std": round(np.std(r2_scores), 4),
        })
    except Exception as e:
        print(f"  HistGBR/{fs_name} failed: {e}")

    # Ridge (needs scaling, no NaN)
    try:
        ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        mae_scores = []
        r2_scores = []
        for tr_idx, te_idx in gkf.split(X_use, groups=groups_use):
            ridge_pipe.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
            pred = ridge_pipe.predict(X_use.iloc[te_idx])
            mae_scores.append(mean_absolute_error(y_use.iloc[te_idx], pred))
            r2_scores.append(r2_score(y_use.iloc[te_idx], pred))
        reg_baseline_results.append({
            "feature_set": fs_name,
            "model": "Ridge",
            "n_features": X_use.shape[1],
            "MAE_mean": round(np.mean(mae_scores), 4),
            "MAE_std": round(np.std(mae_scores), 4),
            "R2_mean": round(np.mean(r2_scores), 4),
            "R2_std": round(np.std(r2_scores), 4),
        })
    except Exception as e:
        print(f"  Ridge/{fs_name} failed: {e}")

reg_baseline_df = pd.DataFrame(reg_baseline_results)
if len(reg_baseline_df) > 0:
    print("\n  Regression baseline results:")
    print(reg_baseline_df.sort_values("MAE_mean").to_string(index=False))

# --- Classification baselines ---
print("\n  Classification baselines …")
cls_feature_sets = {
    "raw_operating": OPERATING,
    "raw_sensors": SENSORS,
    "all_raw": NUMERIC,
    "ops_plus_electrical": OPERATING + electrical_feats,
    "sensors_plus_consistency": SENSORS + sensor_agg_feats + sensor_consist,
    "all_engineered": NUMERIC + all_eng_cols,
    "no_S4": no_s4_raw + no_s4_eng,
    "top_compact": top_cls_feats,
}

cls_baseline_results = []

for fs_name, fs_cols in cls_feature_sets.items():
    X = prepare_X(fs_cols)
    X_use = X.fillna(X.median())
    y_use = y_cls
    groups_use = train["_cv_group"]

    if X_use.shape[1] == 0:
        continue

    # HistGradientBoosting
    hgb_c = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
    try:
        f1_scores = []
        bacc_scores = []
        auc_scores = []
        mcc_scores = []
        for tr_idx, te_idx in gkf.split(X_use, groups=groups_use):
            hgb_c.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
            pred = hgb_c.predict(X_use.iloc[te_idx])
            pred_proba = hgb_c.predict_proba(X_use.iloc[te_idx])[:, 1]
            f1_scores.append(f1_score(y_use.iloc[te_idx], pred, pos_label=1))
            bacc_scores.append(balanced_accuracy_score(y_use.iloc[te_idx], pred))
            try:
                auc_scores.append(roc_auc_score(y_use.iloc[te_idx], pred_proba))
            except:
                pass
            mcc_scores.append(matthews_corrcoef(y_use.iloc[te_idx], pred))
        cls_baseline_results.append({
            "feature_set": fs_name,
            "model": "HistGBC",
            "n_features": X_use.shape[1],
            "F1_Invalid_mean": round(np.mean(f1_scores), 4),
            "F1_Invalid_std": round(np.std(f1_scores), 4),
            "BalAcc_mean": round(np.mean(bacc_scores), 4),
            "ROC_AUC_mean": round(np.mean(auc_scores), 4) if auc_scores else np.nan,
            "MCC_mean": round(np.mean(mcc_scores), 4),
        })
    except Exception as e:
        print(f"  HistGBC/{fs_name} failed: {e}")

    # Logistic Regression
    try:
        lr_pipe = Pipeline([("scaler", StandardScaler()),
                            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
        f1_scores = []
        bacc_scores = []
        auc_scores = []
        for tr_idx, te_idx in gkf.split(X_use, groups=groups_use):
            lr_pipe.fit(X_use.iloc[tr_idx], y_use.iloc[tr_idx])
            pred = lr_pipe.predict(X_use.iloc[te_idx])
            pred_proba = lr_pipe.predict_proba(X_use.iloc[te_idx])[:, 1]
            f1_scores.append(f1_score(y_use.iloc[te_idx], pred, pos_label=1))
            bacc_scores.append(balanced_accuracy_score(y_use.iloc[te_idx], pred))
            try:
                auc_scores.append(roc_auc_score(y_use.iloc[te_idx], pred_proba))
            except:
                pass
        cls_baseline_results.append({
            "feature_set": fs_name,
            "model": "LogReg",
            "n_features": X_use.shape[1],
            "F1_Invalid_mean": round(np.mean(f1_scores), 4),
            "F1_Invalid_std": round(np.std(f1_scores), 4),
            "BalAcc_mean": round(np.mean(bacc_scores), 4),
            "ROC_AUC_mean": round(np.mean(auc_scores), 4) if auc_scores else np.nan,
            "MCC_mean": np.nan,
        })
    except Exception as e:
        print(f"  LogReg/{fs_name} failed: {e}")

cls_baseline_df = pd.DataFrame(cls_baseline_results)
if len(cls_baseline_df) > 0:
    print("\n  Classification baseline results:")
    print(cls_baseline_df.sort_values("F1_Invalid_mean", ascending=False).to_string(index=False))

train.drop(columns=["_cv_group"], inplace=True)

# ╔══════════════════════════════════════════╗
# ║  SECTION 8 & 9 – MODEL RECOMMENDATIONS  ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTIONS 8 & 9 – MODEL RECOMMENDATIONS")
print("="*80)

# Classification recommendations
cls_models = [
    {
        "model": "Logistic Regression (class-weighted)",
        "role": "Baseline",
        "feature_groups": "All raw + electrical features (scaled)",
        "missing_handling": "Impute (median) + missing indicators",
        "scaling": "Required (StandardScaler)",
        "class_imbalance": "class_weight='balanced'",
        "key_hyperparams": "C, penalty (l1/l2/elasticnet), solver",
        "advantages": "Interpretable coefficients, fast, stable baseline",
        "risks": "Cannot capture nonlinear sensor interactions",
        "train_time": "Very fast (<1s)",
    },
    {
        "model": "Random Forest Classifier",
        "role": "Primary candidate",
        "feature_groups": "All raw + engineered features",
        "missing_handling": "Impute median or use surrogate splits",
        "scaling": "Not required",
        "class_imbalance": "class_weight='balanced_subsample'",
        "key_hyperparams": "n_estimators, max_depth, min_samples_leaf, max_features",
        "advantages": "Robust to outliers, handles interactions, feature importance",
        "risks": "Can overfit on small minority class, biased importance for high-cardinality",
        "train_time": "Fast (~2-5s)",
    },
    {
        "model": "Extra Trees Classifier",
        "role": "Primary candidate",
        "feature_groups": "All raw + sensor consistency features",
        "missing_handling": "Impute median",
        "scaling": "Not required",
        "class_imbalance": "class_weight='balanced'",
        "key_hyperparams": "n_estimators, max_depth, min_samples_leaf",
        "advantages": "More random splits reduce overfitting, often better than RF on noisy data",
        "risks": "Slightly less interpretable than RF",
        "train_time": "Fast (~2-5s)",
    },
    {
        "model": "HistGradientBoostingClassifier",
        "role": "Primary candidate",
        "feature_groups": "All features including raw NaN",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "class_imbalance": "sample_weight or SMOTE",
        "key_hyperparams": "max_iter, max_depth, learning_rate, min_samples_leaf, l2_regularization",
        "advantages": "Native NaN, fast, handles interactions well",
        "risks": "No built-in class_weight; needs manual sample weighting",
        "train_time": "Fast (~3-8s)",
    },
    {
        "model": "CatBoost Classifier",
        "role": "Primary candidate",
        "feature_groups": "All raw features (handles NaN/cats natively)",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "class_imbalance": "auto_class_weights='Balanced'",
        "key_hyperparams": "iterations, depth, learning_rate, l2_leaf_reg",
        "advantages": "Excellent default performance, ordered boosting reduces overfitting",
        "risks": "Slower than LightGBM, large model size",
        "train_time": "Moderate (~10-20s)",
    },
    {
        "model": "XGBoost Classifier",
        "role": "Primary candidate",
        "feature_groups": "All engineered features",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "class_imbalance": "scale_pos_weight",
        "key_hyperparams": "n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha/lambda",
        "advantages": "Well-proven, flexible regularization",
        "risks": "Requires careful tuning to avoid overfitting on 1000 rows",
        "train_time": "Fast (~5-10s)",
    },
    {
        "model": "LightGBM Classifier",
        "role": "Primary candidate",
        "feature_groups": "All features",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "class_imbalance": "is_unbalance=True or scale_pos_weight",
        "key_hyperparams": "n_estimators, num_leaves, learning_rate, min_child_samples, reg_alpha/lambda",
        "advantages": "Fastest GBM, leaf-wise growth can be more accurate",
        "risks": "Can overfit with too many leaves on small data",
        "train_time": "Very fast (~2-5s)",
    },
    {
        "model": "Calibrated Stacking Ensemble",
        "role": "Ensemble candidate",
        "feature_groups": "Meta-features from base models",
        "missing_handling": "Delegated to base models",
        "scaling": "Depends on meta-learner",
        "class_imbalance": "Handled by base models",
        "key_hyperparams": "Base model selection, meta-learner, calibration method",
        "advantages": "Combines complementary strengths, often best overall",
        "risks": "Complexity, risk of overfitting meta-learner on small data",
        "train_time": "Moderate (~30-60s)",
    },
]

cls_models_df = pd.DataFrame(cls_models)
cls_models_df.to_csv(OUT / "model_recommendations_classification.csv", index=False)
print("  Classification models saved")

# Regression recommendations
reg_models = [
    {
        "model": "Ridge Regression",
        "role": "Baseline",
        "feature_groups": "Raw + electrical features (scaled)",
        "missing_handling": "Impute median + indicators",
        "scaling": "Required",
        "key_hyperparams": "alpha",
        "sensitivity_outliers": "High – affected by sensor spikes",
        "interpretability": "High – coefficient interpretation",
        "advantages": "Fast, interpretable, stable",
        "risks": "Cannot capture nonlinear relationships",
        "train_time": "Very fast (<1s)",
    },
    {
        "model": "Elastic Net Regression",
        "role": "Baseline",
        "feature_groups": "All engineered features (scaled)",
        "missing_handling": "Impute median + indicators",
        "scaling": "Required",
        "key_hyperparams": "alpha, l1_ratio",
        "sensitivity_outliers": "High",
        "interpretability": "High – sparse coefficients",
        "advantages": "Feature selection via L1, handles multicollinearity",
        "risks": "Linear only",
        "train_time": "Very fast (<1s)",
    },
    {
        "model": "Huber Regressor",
        "role": "Baseline",
        "feature_groups": "Raw + electrical features (scaled)",
        "missing_handling": "Impute median",
        "scaling": "Required",
        "key_hyperparams": "epsilon, alpha",
        "sensitivity_outliers": "Low – robust to outliers",
        "interpretability": "High",
        "advantages": "Robust to sensor spikes, interpretable",
        "risks": "Linear only, less accurate if relationships are nonlinear",
        "train_time": "Very fast (<1s)",
    },
    {
        "model": "Random Forest Regressor",
        "role": "Primary candidate",
        "feature_groups": "All raw + engineered features",
        "missing_handling": "Impute median",
        "scaling": "Not required",
        "key_hyperparams": "n_estimators, max_depth, min_samples_leaf, max_features",
        "sensitivity_outliers": "Moderate – tree splits are robust",
        "interpretability": "Moderate – feature importance",
        "advantages": "Captures interactions, robust",
        "risks": "Can overfit, doesn't extrapolate well",
        "train_time": "Fast (~3-5s)",
    },
    {
        "model": "Extra Trees Regressor",
        "role": "Primary candidate",
        "feature_groups": "All raw + sensor features",
        "missing_handling": "Impute median",
        "scaling": "Not required",
        "key_hyperparams": "n_estimators, max_depth, min_samples_leaf",
        "sensitivity_outliers": "Moderate",
        "interpretability": "Moderate",
        "advantages": "More diverse than RF, faster training",
        "risks": "Similar to RF risks",
        "train_time": "Fast (~3-5s)",
    },
    {
        "model": "HistGradientBoostingRegressor",
        "role": "Primary candidate",
        "feature_groups": "All features including raw NaN",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "key_hyperparams": "max_iter, max_depth, learning_rate, l2_regularization",
        "sensitivity_outliers": "Moderate – MAE loss option",
        "interpretability": "Low-moderate",
        "advantages": "Native NaN, fast, handles nonlinearity",
        "risks": "Needs tuning to avoid overfitting",
        "train_time": "Fast (~5-8s)",
    },
    {
        "model": "CatBoost Regressor",
        "role": "Primary candidate",
        "feature_groups": "All raw features",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "key_hyperparams": "iterations, depth, learning_rate, l2_leaf_reg",
        "sensitivity_outliers": "Moderate – MAE/Quantile loss options",
        "interpretability": "Low-moderate – SHAP available",
        "advantages": "Excellent defaults, ordered boosting",
        "risks": "Slower",
        "train_time": "Moderate (~10-20s)",
    },
    {
        "model": "XGBoost Regressor",
        "role": "Primary candidate",
        "feature_groups": "All engineered features",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "key_hyperparams": "n_estimators, max_depth, learning_rate, subsample, colsample_bytree",
        "sensitivity_outliers": "Moderate",
        "interpretability": "Low-moderate",
        "advantages": "Flexible, well-proven",
        "risks": "Requires tuning",
        "train_time": "Fast (~5-10s)",
    },
    {
        "model": "LightGBM Regressor",
        "role": "Primary candidate",
        "feature_groups": "All features",
        "missing_handling": "Native NaN support",
        "scaling": "Not required",
        "key_hyperparams": "n_estimators, num_leaves, learning_rate, min_child_samples",
        "sensitivity_outliers": "Moderate",
        "interpretability": "Low-moderate",
        "advantages": "Fastest GBM, good accuracy",
        "risks": "Can overfit with many leaves",
        "train_time": "Very fast (~2-5s)",
    },
    {
        "model": "Weighted Ensemble (Ridge + CatBoost + LightGBM)",
        "role": "Ensemble candidate",
        "feature_groups": "Model-specific optimal sets",
        "missing_handling": "Delegated to base models",
        "scaling": "Delegated",
        "key_hyperparams": "Ensemble weights, base model hyperparameters",
        "sensitivity_outliers": "Low – diversity helps",
        "interpretability": "Low",
        "advantages": "Combines linear + nonlinear strengths",
        "risks": "Complexity, potential overfitting of weights",
        "train_time": "Moderate (~30-60s)",
    },
]

reg_models_df = pd.DataFrame(reg_models)
reg_models_df.to_csv(OUT / "model_recommendations_regression.csv", index=False)
print("  Regression models saved")

# ╔══════════════════════════════════════════╗
# ║  SECTION 10 – FORMULA INVESTIGATION     ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 10 – FORMULA INVESTIGATION")
print("="*80)

# Prepare clean data for formula fitting
formula_data = train[NUMERIC + [TARGET_REG]].dropna()
X_form = formula_data[NUMERIC].values
y_form = formula_data[TARGET_REG].values
feature_names_form = NUMERIC

print(f"  Formula fitting on {len(formula_data)} complete cases")

formula_results = []

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_form, y_form)
y_pred_lr = lr.predict(X_form)
r2_lr = r2_score(y_form, y_pred_lr)
mae_lr = mean_absolute_error(y_form, y_pred_lr)
formula_results.append({"model": "Linear Regression", "R2": round(r2_lr, 4), "MAE": round(mae_lr, 4)})
print(f"\n  Linear Regression: R²={r2_lr:.4f}, MAE={mae_lr:.4f}")
print(f"    Coefficients: {dict(zip(feature_names_form, np.round(lr.coef_, 4)))}")
print(f"    Intercept: {lr.intercept_:.4f}")

# 2. Ridge Regression
scaler_form = StandardScaler()
X_form_sc = scaler_form.fit_transform(X_form)
ridge = Ridge(alpha=1.0)
ridge.fit(X_form_sc, y_form)
y_pred_ridge = ridge.predict(X_form_sc)
r2_ridge = r2_score(y_form, y_pred_ridge)
mae_ridge = mean_absolute_error(y_form, y_pred_ridge)
formula_results.append({"model": "Ridge Regression", "R2": round(r2_ridge, 4), "MAE": round(mae_ridge, 4)})
print(f"\n  Ridge Regression: R²={r2_ridge:.4f}, MAE={mae_ridge:.4f}")

# 3. Degree-2 Polynomial
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_form)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_form)
y_pred_poly = lr_poly.predict(X_poly)
r2_poly = r2_score(y_form, y_pred_poly)
mae_poly = mean_absolute_error(y_form, y_pred_poly)
formula_results.append({"model": "Poly-2 Regression", "R2": round(r2_poly, 4), "MAE": round(mae_poly, 4)})
print(f"\n  Polynomial (deg-2) Regression: R²={r2_poly:.4f}, MAE={mae_poly:.4f}")
print(f"    {X_poly.shape[1]} polynomial features")

# 4. Huber Regression
huber = HuberRegressor(epsilon=1.35, max_iter=200)
huber.fit(X_form_sc, y_form)
y_pred_huber = huber.predict(X_form_sc)
r2_huber = r2_score(y_form, y_pred_huber)
mae_huber = mean_absolute_error(y_form, y_pred_huber)
formula_results.append({"model": "Huber Regression", "R2": round(r2_huber, 4), "MAE": round(mae_huber, 4)})
print(f"\n  Huber Regression: R²={r2_huber:.4f}, MAE={mae_huber:.4f}")

# 5. Spline (GAM-like via SplineTransformer)
try:
    spline = SplineTransformer(n_knots=5, degree=3)
    X_spline = spline.fit_transform(X_form)
    lr_spline = Ridge(alpha=0.1)
    lr_spline.fit(X_spline, y_form)
    y_pred_spline = lr_spline.predict(X_spline)
    r2_spline = r2_score(y_form, y_pred_spline)
    mae_spline = mean_absolute_error(y_form, y_pred_spline)
    formula_results.append({"model": "Spline + Ridge", "R2": round(r2_spline, 4), "MAE": round(mae_spline, 4)})
    print(f"\n  Spline + Ridge: R²={r2_spline:.4f}, MAE={mae_spline:.4f}")
except Exception as e:
    print(f"\n  Spline + Ridge failed: {e}")

# Compare with tree baseline
hgb_form = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
hgb_form.fit(X_form, y_form)
y_pred_hgb = hgb_form.predict(X_form)
r2_hgb = r2_score(y_form, y_pred_hgb)
mae_hgb = mean_absolute_error(y_form, y_pred_hgb)
formula_results.append({"model": "HistGBR (baseline)", "R2": round(r2_hgb, 4), "MAE": round(mae_hgb, 4)})
print(f"\n  HistGBR (tree baseline): R²={r2_hgb:.4f}, MAE={mae_hgb:.4f}")

formula_df = pd.DataFrame(formula_results)
print("\n  Formula comparison:")
print(formula_df.to_string(index=False))

# Cross-validated formula comparison
print("\n  Cross-validated formula comparison …")
cv_formula_results = []
kf_form = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels_form = train.loc[formula_data.index, TARGET_CLS].values

for name, model in [("Linear", LinearRegression()),
                     ("Ridge", Pipeline([("s", StandardScaler()), ("r", Ridge(1.0))])),
                     ("Huber", Pipeline([("s", StandardScaler()), ("h", HuberRegressor())])),
                     ("HistGBR", HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42))]:
    cv_mae = []
    cv_r2 = []
    for tr_i, te_i in kf_form.split(X_form, labels_form):
        model.fit(X_form[tr_i], y_form[tr_i])
        pred = model.predict(X_form[te_i])
        cv_mae.append(mean_absolute_error(y_form[te_i], pred))
        cv_r2.append(r2_score(y_form[te_i], pred))
    cv_formula_results.append({
        "model": name,
        "CV_MAE_mean": round(np.mean(cv_mae), 4),
        "CV_MAE_std": round(np.std(cv_mae), 4),
        "CV_R2_mean": round(np.mean(cv_r2), 4),
        "CV_R2_std": round(np.std(cv_r2), 4),
    })

cv_formula_df = pd.DataFrame(cv_formula_results)
print(cv_formula_df.to_string(index=False))

# ╔══════════════════════════════════════════╗
# ║  SECTION 11 – EXPERIMENT MATRIX         ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 11 – EXPERIMENT MATRIX")
print("="*80)

experiments = [
    {"exp": 1, "task": "Classification", "model": "Logistic Regression", "features": "Raw operating + electrical", "missing": "Median impute + indicators", "scaling": "StandardScaler", "class_weight": "balanced", "key_hp": "C, penalty", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Interpretable baseline for classification"},
    {"exp": 2, "task": "Classification", "model": "Random Forest", "features": "All raw + engineered", "missing": "Median impute", "scaling": "None", "class_weight": "balanced_subsample", "key_hp": "n_estimators, max_depth", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Strong nonlinear baseline"},
    {"exp": 3, "task": "Classification", "model": "Extra Trees", "features": "All raw + sensor consistency", "missing": "Median impute", "scaling": "None", "class_weight": "balanced", "key_hp": "n_estimators, max_depth", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "More randomized than RF, may reduce overfitting"},
    {"exp": 4, "task": "Classification", "model": "HistGBClassifier", "features": "All features (raw NaN)", "missing": "Native", "scaling": "None", "class_weight": "sample_weight", "key_hp": "max_iter, lr, max_depth", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Native NaN handling, no imputation needed"},
    {"exp": 5, "task": "Classification", "model": "CatBoost", "features": "Raw features + missing indicators", "missing": "Native", "scaling": "None", "class_weight": "Balanced", "key_hp": "iterations, depth, lr", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Excellent defaults, ordered boosting"},
    {"exp": 6, "task": "Classification", "model": "XGBoost", "features": "All engineered features", "missing": "Native", "scaling": "None", "class_weight": "scale_pos_weight", "key_hp": "n_est, max_depth, lr, subsample", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Flexible regularization for small dataset"},
    {"exp": 7, "task": "Classification", "model": "LightGBM", "features": "All features", "missing": "Native", "scaling": "None", "class_weight": "is_unbalance", "key_hp": "num_leaves, lr, min_child", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Fastest GBM, leaf-wise growth"},
    {"exp": 8, "task": "Classification", "model": "CatBoost", "features": "Sensor consistency only", "missing": "Native", "scaling": "None", "class_weight": "Balanced", "key_hp": "iterations, depth", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Test if sensor consistency alone predicts validity"},
    {"exp": 9, "task": "Classification", "model": "Stacking (LR + RF + XGB)", "features": "Top compact features", "missing": "Mixed", "scaling": "Per-model", "class_weight": "Per-model", "key_hp": "Meta-learner, base hp", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Combine linear + nonlinear strengths"},
    {"exp": 10, "task": "Classification", "model": "Soft Voting (CatB + LGBM + HGB)", "features": "All features", "missing": "Native", "scaling": "None", "class_weight": "Per-model", "key_hp": "voting weights", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Diverse boosting ensemble"},
    {"exp": 11, "task": "Regression", "model": "Ridge", "features": "Raw + electrical features", "missing": "Median impute", "scaling": "StandardScaler", "class_weight": "N/A", "key_hp": "alpha", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Linear baseline for regression"},
    {"exp": 12, "task": "Regression", "model": "Elastic Net", "features": "All engineered (scaled)", "missing": "Median impute + indicators", "scaling": "StandardScaler", "class_weight": "N/A", "key_hp": "alpha, l1_ratio", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Sparse feature selection via L1"},
    {"exp": 13, "task": "Regression", "model": "Huber", "features": "Raw + electrical features", "missing": "Median impute", "scaling": "StandardScaler", "class_weight": "N/A", "key_hp": "epsilon, alpha", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Robust to sensor outliers"},
    {"exp": 14, "task": "Regression", "model": "Random Forest", "features": "All raw + engineered", "missing": "Median impute", "scaling": "None", "class_weight": "N/A", "key_hp": "n_estimators, max_depth", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Nonlinear baseline for regression"},
    {"exp": 15, "task": "Regression", "model": "HistGBRegressor", "features": "All features (raw NaN)", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "max_iter, lr, max_depth", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Native NaN handling for regression"},
    {"exp": 16, "task": "Regression", "model": "CatBoost", "features": "Raw features", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "iterations, depth, lr", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Excellent defaults for regression"},
    {"exp": 17, "task": "Regression", "model": "XGBoost", "features": "All engineered features", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "n_est, max_depth, lr, subsample", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Flexible nonlinear regression"},
    {"exp": 18, "task": "Regression", "model": "LightGBM", "features": "All features", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "num_leaves, lr, min_child", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Fast GBM regression"},
    {"exp": 19, "task": "Regression", "model": "Ridge", "features": "Operating features only (sensor-free)", "missing": "None needed", "scaling": "StandardScaler", "class_weight": "N/A", "key_hp": "alpha", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Test regression without any sensor data"},
    {"exp": 20, "task": "Regression", "model": "CatBoost", "features": "Sensors + sensor consistency only", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "iterations, depth", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Test regression with sensor-heavy features only"},
    {"exp": 21, "task": "Regression", "model": "Poly-2 + Ridge", "features": "Raw features (polynomial)", "missing": "Median impute", "scaling": "StandardScaler", "class_weight": "N/A", "key_hp": "alpha", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Interpretable polynomial formula"},
    {"exp": 22, "task": "Regression", "model": "Weighted Avg (Ridge+CatB+LGBM)", "features": "Model-specific", "missing": "Mixed", "scaling": "Per-model", "class_weight": "N/A", "key_hp": "ensemble weights", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Linear + nonlinear ensemble"},
    {"exp": 23, "task": "Regression", "model": "HistGBRegressor", "features": "Valid records only", "missing": "Native", "scaling": "None", "class_weight": "N/A", "key_hp": "max_iter, lr, max_depth", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Test if excluding Invalid improves regression"},
    {"exp": 24, "task": "Regression", "model": "Extra Trees", "features": "Top compact features", "missing": "Median impute", "scaling": "None", "class_weight": "N/A", "key_hp": "n_estimators, max_depth", "validation": "GroupKFold(5)", "metrics": "MAE, RMSE, R², MedAE", "reason": "Compact feature set with diverse model"},
    {"exp": 25, "task": "Classification", "model": "HistGBClassifier", "features": "Top compact features", "missing": "Native", "scaling": "None", "class_weight": "sample_weight", "key_hp": "max_iter, lr", "validation": "GroupKFold(5)", "metrics": "F1, BalAcc, ROC-AUC, MCC", "reason": "Minimal feature set with strong model"},
]

exp_df = pd.DataFrame(experiments)
exp_df.to_csv(OUT / "experiment_matrix.csv", index=False)
print(f"  Experiment matrix: {len(exp_df)} experiments saved")

# ╔══════════════════════════════════════════╗
# ║  SECTION 12 – DATA QUALITY CSV          ║
# ╚══════════════════════════════════════════╝
print("\n" + "="*80)
print("SECTION 12 – GENERATING OUTPUTS")
print("="*80)

# data_quality_findings.csv
dq_rows = []
for k, v in findings.items():
    dq_rows.append({"finding": k, "value": str(v)})
dq_df = pd.DataFrame(dq_rows)
dq_df.to_csv(OUT / "data_quality_findings.csv", index=False)
print("  data_quality_findings.csv saved")

# relationship_report.md
print("  Writing relationship_report.md …")
with open(OUT / "relationship_report.md", "w", encoding="utf-8") as f:
    f.write("# Mathematical Relationship Analysis Report\n\n")
    f.write(f"Generated: {datetime.now()}\n\n")

    f.write("## Regression Relationships (Feature → Reference_Parameter)\n\n")
    f.write("| Feature | Pearson r | Spearman r | Linear R² | Poly2 R² | MI | Partial r |\n")
    f.write("|---------|-----------|------------|-----------|----------|----|-----------|\n")
    for feat, vals in reg_analysis.items():
        f.write(f"| {feat} | {vals.get('pearson_r', '')} | {vals.get('spearman_r', '')} | "
                f"{vals.get('linear_R2', '')} | {vals.get('poly2_R2', '')} | "
                f"{vals.get('mutual_info', '')} | {vals.get('partial_corr_ctrl_ops', '')} |\n")

    f.write("\n## Classification Relationships (Feature → Validity_Label)\n\n")
    f.write("| Feature | Point-biserial r | Cohen's d | AUC | MI |\n")
    f.write("|---------|------------------|-----------|-----|----|\n")
    for feat, vals in cls_analysis.items():
        f.write(f"| {feat} | {vals.get('point_biserial_r', '')} | {vals.get('cohens_d', '')} | "
                f"{vals.get('univariate_auc', '')} | {vals.get('mutual_info', '')} |\n")

    f.write("\n## Bootstrap 95% CIs (Pearson → Ref Param)\n\n")
    for feat, vals in reg_analysis.items():
        ci_lo = vals.get('pearson_ci_lo', '')
        ci_hi = vals.get('pearson_ci_hi', '')
        if ci_lo:
            f.write(f"- **{feat}**: [{ci_lo}, {ci_hi}]\n")

    f.write("\n## VIF (Multicollinearity)\n\n")
    if "VIF" in findings:
        for v in findings["VIF"]:
            f.write(f"- **{v['feature']}**: VIF = {v['VIF']}\n")

    f.write("\n## Formula Investigation\n\n")
    f.write("| Model | In-sample R² | In-sample MAE |\n")
    f.write("|-------|-------------|---------------|\n")
    for row in formula_results:
        f.write(f"| {row['model']} | {row['R2']} | {row['MAE']} |\n")

    if len(lr.coef_) > 0:
        f.write("\n### Linear Regression Formula\n\n")
        f.write(f"Reference_Parameter = {lr.intercept_:.4f}")
        for name, coef in zip(feature_names_form, lr.coef_):
            f.write(f" + ({coef:.4f}) × {name}")
        f.write("\n")

    f.write("\n### Cross-Validated Formula Comparison\n\n")
    f.write(cv_formula_df.to_markdown(index=False))
    f.write("\n")

print("  relationship_report.md saved")

# ╔══════════════════════════════════════════════════════╗
# ║  GENERATE eda_report.html                           ║
# ╚══════════════════════════════════════════════════════╝
print("  Generating eda_report.html …")

# Collect chart files
chart_files = sorted(CHARTS.glob("*.png"))

html_parts = []
html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CPRI Hackathon – EDA Report</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f8f9fa; color: #333; }
h1, h2, h3 { color: #2c3e50; }
.section { background: #fff; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }
th { background: #3498db; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
img { max-width: 100%; height: auto; margin: 10px 0; border-radius: 4px; }
.highlight { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }
.danger { background: #f8d7da; padding: 10px; border-left: 4px solid #dc3545; margin: 10px 0; }
.success { background: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }
.info { background: #d1ecf1; padding: 10px; border-left: 4px solid #17a2b8; margin: 10px 0; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
.chart-single { text-align: center; }
</style>
</head>
<body>
""")

html_parts.append("<h1>CPRI Hackathon – Exploration & Model-Planning Report</h1>")
html_parts.append(f"<p><em>Generated: {datetime.now()}</em></p>")

# Executive Summary
html_parts.append('<div class="section"><h2>1. Executive Summary</h2>')
html_parts.append(f"""
<ul>
<li><strong>Training data:</strong> {train.shape[0]} rows × {train.shape[1]} columns</li>
<li><strong>Test data:</strong> {test.shape[0]} rows × {test.shape[1] if hasattr(test, 'shape') else 'N/A'} columns</li>
<li><strong>Class balance:</strong> {vc.to_dict()}</li>
<li><strong>Missing values:</strong> Concentrated in sensor columns; Sensor_S4 has the most</li>
<li><strong>Duplicates:</strong> {n_dup_meas} rows in duplicate measurement groups</li>
<li><strong>Engineered features:</strong> {train_feat.shape[1]} features created</li>
</ul>
<div class="highlight">
<strong>Key Findings:</strong>
<ul>
<li>Moderate class imbalance (~{(vc.get('Invalid', 0)/len(train)*100):.0f}% Invalid) – use class-weighted models</li>
<li>Sensor missingness is informative – create missing indicators</li>
<li>Duplicate measurement vectors with conflicting targets exist – keep groups in same CV fold</li>
<li>Sensor consistency features (residuals, deviations) are promising for classification</li>
<li>Tree-based models significantly outperform linear models for both tasks</li>
</ul>
</div>
</div>
""")

# Data Quality
html_parts.append('<div class="section"><h2>2. Data Quality Findings</h2>')
html_parts.append(f"""
<h3>Missing Values</h3>
<table>
<tr><th>Column</th><th>Train Missing</th><th>Train %</th><th>Test Missing</th></tr>
""")
for c in train.columns:
    tm = int(miss_train.get(c, 0))
    tp = round(miss_pct_train.get(c, 0), 2)
    tst = int(miss_test.get(c, 0)) if c in miss_test else "N/A"
    if tm > 0 or (isinstance(tst, int) and tst > 0):
        html_parts.append(f"<tr><td>{c}</td><td>{tm}</td><td>{tp}%</td><td>{tst}</td></tr>")
html_parts.append("</table>")

html_parts.append(f"""
<h3>Duplicate Analysis</h3>
<ul>
<li>Exact full-row duplicates: {findings.get('exact_full_row_dups', 0)}</li>
<li>Duplicate measurement vectors: {findings.get('dup_measurement_vectors', 0)}</li>
<li>Number of duplicate groups: {findings.get('dup_measurement_groups', 0)}</li>
<li>Groups with conflicting Reference_Parameter: {findings.get('dup_conflicting_ref_param', 0)}</li>
<li>Groups with conflicting Validity_Label: {findings.get('dup_conflicting_validity', 0)}</li>
<li>Validity of duplicate rows: {findings.get('dup_labels', {})}</li>
<li>Test duplicate measurement vectors: {findings.get('test_dup_measurement_vectors', 0)}</li>
</ul>
<div class="danger">
<strong>Duplicate handling recommendation:</strong> Do not remove duplicates before modelling. 
Instead, create a duplicate-group ID and ensure all members of a duplicate group stay in the same 
CV fold (GroupKFold). Create duplicate-flag and group-size features. Investigate whether 
conflicting targets are data-entry errors or legitimate re-measurements.
</div>
""")

html_parts.append(f"""
<h3>Extreme Values</h3>
<table>
<tr><th>Feature</th><th>IQR Outliers</th><th>Robust-Z Outliers</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>
""")
for c, v in extreme_summary.items():
    html_parts.append(f"<tr><td>{c}</td><td>{v['iqr_outliers']}</td><td>{v['robust_z_outliers']}</td>"
                      f"<td>{v['min']:.2f}</td><td>{v['max']:.2f}</td><td>{v['mean']:.2f}</td><td>{v['std']:.2f}</td></tr>")
html_parts.append("</table>")

html_parts.append(f"""
<h3>Train vs Test Distribution (KS Tests)</h3>
<table>
<tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Significant?</th></tr>
""")
for c, v in ks_results.items():
    sig = "⚠️ YES" if v['p_value'] < 0.05 else "No"
    html_parts.append(f"<tr><td>{c}</td><td>{v['statistic']}</td><td>{v['p_value']}</td><td>{sig}</td></tr>")
html_parts.append("</table>")
html_parts.append("</div>")

# Charts section
html_parts.append('<div class="section"><h2>3. Exploratory Charts</h2>')
for cf in chart_files:
    chart_name = cf.stem.replace("_", " ").title()
    interp_key = cf.stem
    html_parts.append(f'<h3>{chart_name}</h3>')
    html_parts.append(f'<div class="chart-single"><img src="charts/{cf.name}" alt="{chart_name}"></div>')
    if interp_key in interpretations:
        html_parts.append(f'<div class="info">{interpretations[interp_key]}</div>')
html_parts.append("</div>")

# Relationships
html_parts.append('<div class="section"><h2>4. Mathematical Relationships</h2>')
html_parts.append("<h3>Regression Feature Analysis</h3>")
# Drop columns with dict values before rendering to HTML
reg_html_df = reg_analysis_df.select_dtypes(include=[np.number]).copy()
html_parts.append(reg_html_df.round(4).to_html(classes=""))
html_parts.append("<h3>Classification Feature Analysis</h3>")
cls_html_df = cls_analysis_df.copy()
for col in cls_html_df.columns:
    if cls_html_df[col].apply(lambda x: isinstance(x, dict)).any():
        cls_html_df = cls_html_df.drop(columns=[col])
cls_html_df = cls_html_df.select_dtypes(include=[np.number])
html_parts.append(cls_html_df.round(4).to_html(classes=""))
html_parts.append("</div>")

# Feature Rankings
html_parts.append('<div class="section"><h2>5. Feature Rankings</h2>')
html_parts.append("<h3>Top 25 Regression Features</h3>")
html_parts.append(reg_ranking.head(25).select_dtypes(include=[np.number]).round(4).to_html(classes=""))
html_parts.append("<h3>Top 25 Classification Features</h3>")
html_parts.append(cls_ranking.head(25).select_dtypes(include=[np.number]).round(4).to_html(classes=""))
html_parts.append("</div>")

# Formula
html_parts.append('<div class="section"><h2>6. Formula Investigation</h2>')
html_parts.append(formula_df.to_html(classes="", index=False))
html_parts.append("<h3>Cross-Validated Comparison</h3>")
html_parts.append(cv_formula_df.to_html(classes="", index=False))
if len(lr.coef_) > 0:
    formula_str = f"Reference_Parameter = {lr.intercept_:.4f}"
    for name, coef in zip(feature_names_form, lr.coef_):
        formula_str += f" + ({coef:.4f}) × {name}"
    html_parts.append(f'<div class="highlight"><strong>Linear Formula:</strong><br><code>{formula_str}</code></div>')
html_parts.append("</div>")

# Baselines
html_parts.append('<div class="section"><h2>7. Baseline Experiment Results</h2>')
html_parts.append("<h3>Regression Baselines</h3>")
if len(reg_baseline_df) > 0:
    html_parts.append(reg_baseline_df.sort_values("MAE_mean").to_html(classes="", index=False))
html_parts.append("<h3>Classification Baselines</h3>")
if len(cls_baseline_df) > 0:
    html_parts.append(cls_baseline_df.sort_values("F1_Invalid_mean", ascending=False).to_html(classes="", index=False))
html_parts.append("</div>")

# Model recommendations
html_parts.append('<div class="section"><h2>8. Model Recommendations</h2>')
html_parts.append("<h3>Classification Models</h3>")
html_parts.append(cls_models_df.to_html(classes="", index=False))
html_parts.append("<h3>Regression Models</h3>")
html_parts.append(reg_models_df.to_html(classes="", index=False))
html_parts.append("</div>")

# Experiment matrix
html_parts.append('<div class="section"><h2>9. Experiment Matrix</h2>')
html_parts.append(exp_df.to_html(classes="", index=False))
html_parts.append("</div>")

# Validation strategy
html_parts.append("""<div class="section"><h2>10. Validation Strategy</h2>
<ul>
<li><strong>GroupKFold(5)</strong>: Duplicate measurement groups must stay in the same fold</li>
<li><strong>Stratification</strong>: Maintain class proportions in each fold where possible</li>
<li><strong>Metrics</strong>: Use F1 (Invalid class), BalAcc, ROC-AUC, MCC for classification; MAE, RMSE, R² for regression</li>
<li><strong>Do NOT use accuracy alone</strong> due to class imbalance</li>
<li><strong>Do NOT use Reference_Parameter for classification</strong> – it won't be available at test time</li>
</ul>
</div>""")

# Risks
html_parts.append("""<div class="section"><h2>11. Risks and Leakage Warnings</h2>
<div class="danger">
<ul>
<li><strong>Target leakage</strong>: Do not use Reference_Parameter as input for Validity_Label prediction</li>
<li><strong>Test_ID leakage</strong>: Do not use Test_ID or its numerical part as a feature</li>
<li><strong>Duplicate leakage</strong>: Duplicate rows must be in the same CV fold to avoid data leakage</li>
<li><strong>Sensor_S4 missingness</strong>: High missingness may cause imputation to inject noise</li>
<li><strong>Small dataset</strong>: Only 1000 rows and 134 Invalid – risk of overfitting, especially with many features</li>
<li><strong>Cross-fitted residuals</strong>: Must be computed via out-of-fold prediction to avoid leakage</li>
</ul>
</div>
</div>""")

# Next steps
html_parts.append("""<div class="section"><h2>12. Next Steps</h2>
<ol>
<li>Decide on duplicate handling: keep, average, or flag</li>
<li>Run the 25 experiments from the experiment matrix</li>
<li>Perform hyperparameter tuning on the top 3-5 models per task</li>
<li>Build calibrated ensemble models</li>
<li>Generate final predictions on Test_Data</li>
<li>Create submission CSV and summary.json</li>
</ol>
</div>""")

html_parts.append("</body></html>")

with open(OUT / "eda_report.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))
print("  eda_report.html saved")

# ╔══════════════════════════════════════════════════════╗
# ║  GENERATE NOTEBOOK STUB                             ║
# ╚══════════════════════════════════════════════════════╝
print("  Generating notebook stub …")
import nbformat
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# 01 – Dataset Exploration\n\nThis notebook documents the CPRI Hackathon EDA process.\nFull analysis was run via `analysis_main.py`."),
    nbformat.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ntrain = pd.read_excel(r'd:\\Powernext\\dataset\\CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx', 'Training_Data')\ntest = pd.read_excel(r'd:\\Powernext\\dataset\\CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx', 'Test_Data')"),
    nbformat.v4.new_code_cell("print(f'Training: {train.shape}')\nprint(f'Test: {test.shape}')\ntrain.head()"),
    nbformat.v4.new_code_cell("train.describe()"),
    nbformat.v4.new_code_cell("train.isnull().sum()"),
    nbformat.v4.new_code_cell("train['Validity_Label'].value_counts().plot(kind='bar')\nplt.title('Class Balance')\nplt.show()"),
    nbformat.v4.new_code_cell("# See analysis/eda_report.html for the full report\n# See analysis/charts/ for all generated charts"),
    nbformat.v4.new_markdown_cell("## Key Findings\n\nSee `analysis/eda_report.html` for the complete exploration report including:\n- Data quality audit\n- 22+ chart types\n- Mathematical relationship analysis\n- Feature engineering catalogue\n- Feature rankings\n- Baseline model experiments\n- Model recommendations\n- 25-experiment matrix for the next phase"),
]
with open(Path(r"d:\Powernext\notebooks\01_dataset_exploration.ipynb"), "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("  Notebook saved")

# ╔══════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                      ║
# ╚══════════════════════════════════════════════════════╝
print("\n" + "="*80)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("="*80)
print(f"""
Generated files:
  analysis/eda_report.html
  analysis/relationship_report.md
  analysis/feature_catalog.csv
  analysis/regression_feature_ranking.csv
  analysis/classification_feature_ranking.csv
  analysis/model_recommendations_classification.csv
  analysis/model_recommendations_regression.csv
  analysis/experiment_matrix.csv
  analysis/data_quality_findings.csv
  analysis/train_descriptive_stats.csv
  analysis/charts/ ({len(chart_files)} charts)
  notebooks/01_dataset_exploration.ipynb
  requirements.txt
""")
print(f"Completed: {datetime.now()}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')  # Silences harmless sklearn convergence warnings

# Load the dataset
# You need to download creditcard.csv from Kaggle first
# Kaggle link: https://www.kaggle.com/mlg-ulb/creditcardfraud
PROJECT_DIR = Path(__file__).resolve().parent
data = pd.read_csv(PROJECT_DIR / 'creditcard.csv')

print("Dataset loaded successfully!")
print(f"Shape: {data.shape}")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Peek at first 5 rows
print("\n--- First 5 rows ---")
print(data.head())

# List all column names
print("\n--- Column names ---")
print(data.columns.tolist())

# Check data types of each column
print("\n--- Data types + non-null count ---")
print(data.info())

# Check for missing values in each column
print("\n--- Missing values per column ---")
missing = data.isnull().sum()
print(missing)
print(f"\nTotal missing values in entire dataset: {missing.sum()}")

# Check class distribution (Fraud vs Non-Fraud)
print("\n--- Class Distribution ---")
class_counts = data['Class'].value_counts()
print(f"Non-Fraud (Class 0): {class_counts[0]} transactions")
print(f"Fraud     (Class 1): {class_counts[1]} transactions")

total = data.shape[0]
percent_fraud = (class_counts[1] / total) * 100
percent_normal = (class_counts[0] / total) * 100
print(f"\nNon-Fraud: {percent_normal:.2f}%")
print(f"Fraud:     {percent_fraud:.2f}%")

# Compare transaction amounts: Fraud vs Non-Fraud
print("\n--- Transaction Amount Comparison ---")
non_fraud = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

print("Non-Fraud transactions:")
print(f"  Mean amount:   ${non_fraud['Amount'].mean():.2f}")
print(f"  Median amount: ${non_fraud['Amount'].median():.2f}")
print(f"  Max amount:    ${non_fraud['Amount'].max():.2f}")

print("\nFraud transactions:")
print(f"  Mean amount:   ${fraud['Amount'].mean():.2f}")
print(f"  Median amount: ${fraud['Amount'].median():.2f}")
print(f"  Max amount:    ${fraud['Amount'].max():.2f}")

# Plot 1: Class distribution bar chart
plt.figure(figsize=(8, 5))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution: Fraud (1) vs Non-Fraud (0)')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Number of Transactions')
plt.savefig(PROJECT_DIR / 'class_distribution.png', dpi=100)
plt.close()
print("Chart saved: class_distribution.png")

# Plot 2: Side-by-side histograms of transaction amounts
# We cap at $500 because very large amounts are rare and squash the detail
amount_cap = 500

plt.figure(figsize=(14, 5))

# Left subplot: Non-Fraud
plt.subplot(1, 2, 1)
plt.hist(non_fraud['Amount'], bins=50, range=(0, amount_cap), color='blue', edgecolor='black')
plt.title('Non-Fraud Transaction Amounts')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')

# Right subplot: Fraud
plt.subplot(1, 2, 2)
plt.hist(fraud['Amount'], bins=50, range=(0, amount_cap), color='red', edgecolor='black')
plt.title('Fraud Transaction Amounts')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')

plt.tight_layout()
plt.savefig(PROJECT_DIR / 'amount_histograms.png', dpi=100)
plt.close()
print("Chart saved: amount_histograms.png")

# -------------------------------------------------------
# STEP 3: PREPROCESSING & FEATURE SCALING
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING & FEATURE SCALING")
print("=" * 60)

# Split the data into Features (X) and Target (y)
# X = everything except the Class column (the inputs to our model)
# y = only the Class column (the answer we want to predict)
X = data.drop('Class', axis=1)
y = data['Class']

print(f"\nFeatures (X) shape: {X.shape}   <- {X.shape[1]} columns")
print(f"Target (y) shape:   {y.shape}    <- single column")
print(f"Columns in X: {X.columns.tolist()}")

# First set aside a final test set (20%). We never use it to make choices.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split the remaining 80% into training data (64% of all data) and a validation
# set (16%). The validation set is where we choose probability thresholds.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print("\n--- Train/Test Split Results ---")
print(f"Training set:   {X_train.shape[0]} transactions ({100 * len(X_train)/len(X):.0f}%)")
print(f"Validation set: {X_val.shape[0]} transactions ({100 * len(X_val)/len(X):.0f}%)")
print(f"Test set:       {X_test.shape[0]} transactions ({100 * len(X_test)/len(X):.0f}%)")

# Verify class distribution is preserved in both sets (thanks to stratify)
train_fraud_pct = 100 * y_train.sum() / len(y_train)
val_fraud_pct = 100 * y_val.sum() / len(y_val)
test_fraud_pct = 100 * y_test.sum() / len(y_test)
print(f"\nFraud in training set: {y_train.sum()} ({train_fraud_pct:.3f}%)")
print(f"Fraud in validation set: {y_val.sum()} ({val_fraud_pct:.3f}%)")
print(f"Fraud in test set:     {y_test.sum()} ({test_fraud_pct:.3f}%)")

# Feature Scaling with StandardScaler
# RULE: Only FIT on training data, then TRANSFORM training, validation, and test.
# This prevents data leakage from either holdout set.
scaler = StandardScaler()
scaler.fit(X_train)  # Calculate mean and std ONLY from training set

# Apply the scaling transformation
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames (so columns are preserved for readability)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("\n--- Feature Scaling Complete ---")
print("Before scaling (first 3 rows of X_train):")
print(X_train[['Time', 'V1', 'Amount']].head(3))
print("\nAfter scaling (first 3 rows of X_train_scaled):")
print(X_train_scaled[['Time', 'V1', 'Amount']].head(3))

# Verify scaling worked: each column should have mean ≈ 0, std ≈ 1
print("\nScaling verification (training set):")
print(X_train_scaled[['Time', 'Amount', 'V1']].agg(['mean', 'std']).round(4))

# -------------------------------------------------------
# STEP 4: HANDLING CLASS IMBALANCE
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: HANDLING CLASS IMBALANCE")
print("=" * 60)

# We will create 4 versions of the training data for comparison:
#   1. Original (imbalanced) - baseline
#   2. Random Undersampling
#   3. Random Oversampling
#   4. SMOTE (Synthetic Minority Oversampling)

# ---- 1. ORIGINAL IMBALANCED BASELINE ----
print("\n--- [1/4] Original Imbalanced Training Set ---")
orig_train_fraud = y_train.sum()
orig_train_total = len(y_train)
print(f"  Non-Fraud: {orig_train_total - orig_train_fraud}")
print(f"  Fraud:     {orig_train_fraud}")
print(f"  Ratio:     1 fraud per ~{(orig_train_total - orig_train_fraud)/orig_train_fraud:.0f} non-fraud")

# Keep original as-is (no resampling needed)
X_train_original = X_train_scaled.copy()
y_train_original = y_train.copy()

# ---- 2. RANDOM UNDERSAMPLING ----
print("\n--- [2/4] Random Undersampling ---")
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train_scaled, y_train)

under_fraud = y_train_under.sum()
under_total = len(y_train_under)
print(f"  Non-Fraud: {under_total - under_fraud}")
print(f"  Fraud:     {under_fraud}")
print(f"  Ratio:     1 fraud per {(under_total - under_fraud)//under_fraud} non-fraud (perfectly balanced!)")
print(f"  Dataset shrinkage: {orig_train_total:,} rows -> {under_total:,} rows ({100*under_total/orig_train_total:.1f}% kept)")

# ---- 3. RANDOM OVERSAMPLING ----
print("\n--- [3/4] Random Oversampling ---")
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train_scaled, y_train)

over_fraud = y_train_over.sum()
over_total = len(y_train_over)
print(f"  Non-Fraud: {over_total - over_fraud:,}")
print(f"  Fraud:     {over_fraud:,}")
print(f"  Ratio:     1 fraud per {(over_total - over_fraud)//over_fraud} non-fraud (perfectly balanced!)")
print(f"  Dataset growth: {orig_train_total:,} rows -> {over_total:,} rows ({100*over_total/orig_train_total:.0f}% size)")

# ---- 4. SMOTE ----
print("\n--- [4/4] SMOTE (Synthetic Minority Oversampling) ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

smote_fraud = y_train_smote.sum()
smote_total = len(y_train_smote)
print(f"  Non-Fraud: {smote_total - smote_fraud:,}")
print(f"  Fraud:     {smote_fraud:,}")
print(f"  Ratio:     1 fraud per {(smote_total - smote_fraud)//smote_fraud} non-fraud (perfectly balanced!)")
print(f"  Dataset growth: {orig_train_total:,} rows -> {smote_total:,} rows ({100*smote_total/orig_train_total:.0f}% size)")

# Quick summary of all 4 datasets
print("\n--- Step 4 Summary: Training Set Sizes ---")
print(f"  [1] Original:      {len(X_train_original):>7,} rows  (imbalanced ~577:1)")
print(f"  [2] Undersampled:  {len(X_train_under):>7,} rows  (balanced 1:1, data discarded)")
print(f"  [3] Oversampled:   {len(X_train_over):>7,} rows  (balanced 1:1, fraud duplicated)")
print(f"  [4] SMOTE:         {len(X_train_smote):>7,} rows  (balanced 1:1, fraud synthetic)")
print("\nAll 4 training variants ready -- Step 5: Training 12 models!")

# ======================================
# STEP 5: TRAIN MODELS (4 datasets x 3 types = 12 models)
# ======================================

# Define the 4 training datasets we'll use (stored as a list of tuples)
# Each entry is: (dataset_name, X_train_variant, y_train_variant)
train_variants = [
    ("[1] Original     ", X_train_original, y_train_original),
    ("[2] Undersampled ", X_train_under,    y_train_under),
    ("[3] Oversampled  ", X_train_over,     y_train_over),
    ("[4] SMOTE        ", X_train_smote,    y_train_smote),
]

# Dictionary to store all trained models so we can use them later in Step 6
# Key format: "DatasetName_ModelType"
models = {}

print("\n--- Training Logistic Regression + Random Forest + XGBoost on all 4 datasets ---\n")

for dataset_name, X_variant, y_variant in train_variants:
    name = dataset_name.strip()  # remove padding spaces
    
    # --- MODEL A: Logistic Regression ---
    print(f"Training LogReg on {name}...", end=" ")
    
    logreg = LogisticRegression(
        max_iter=1000,      # More iterations: sometimes the default 100 isn't enough to converge
        random_state=42
    )
    logreg.fit(X_variant, y_variant)  # <--- THE ACTUAL LEARNING HAPPENS HERE
    
    key = f"{name}_LogReg"
    models[key] = logreg
    print("[OK] Done")
    
    # --- MODEL B: Random Forest ---
    print(f"Training RF     on {name}...", end=" ")
    
    rf = RandomForestClassifier(
        n_estimators=100,   # Number of decision trees in the "forest" (100 = good default)
        max_depth=10,       # Each tree can be at most 10 levels deep (prevents overfitting)
        random_state=42,
        n_jobs=-1           # Use ALL CPU cores on your computer (speeds up forest training)
    )
    rf.fit(X_variant, y_variant)  # <--- THE ACTUAL LEARNING HAPPENS HERE
    
    key = f"{name}_RF"
    models[key] = rf
    print("[OK] Done")
    
    # --- MODEL C: XGBoost (Gradient Boosting) ---
    print(f"Training XGB    on {name}...", end=" ")
    
    xgb = XGBClassifier(
        n_estimators=100,    # Number of boosting rounds (trees added one by one)
        max_depth=5,         # Shallow trees (less than RF) to prevent overfitting in boosting
        learning_rate=0.1,   # Step size shrinkage: smaller = slower but more accurate
        n_jobs=-1,           # Use all CPU cores
        random_state=42,
        eval_metric='logloss'  # Avoids deprecation warning for XGBoost
    )
    xgb.fit(X_variant, y_variant)  # <--- THE ACTUAL LEARNING HAPPENS HERE
    
    key = f"{name}_XGB"
    models[key] = xgb
    print("[OK] Done")
    
    print()  # blank line between datasets

print(f"Step 5 complete! {len(models)} supervised models trained and stored in 'models' dict.")
print(f"Model keys: {list(models.keys())}")

# --- UNSUPERVISED MODEL: Isolation Forest (13th model total) ---
# Why unsupervised? It doesn't use ANY fraud labels (y_train) during training!
# Great for when you have zero labeled fraud cases (new product lines, etc.)
print("\n--- Training Isolation Forest (Unsupervised, NO labels used!) ---")
print("Training IF     on [1] Original...", end=" ")

iso_forest = IsolationForest(
    contamination=0.002,  # Expected % of anomalies (we know fraud is ~0.17%, set slightly higher)
    n_estimators=200,     # Number of isolation trees
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
# FIT ONLY on X_train_scaled (NO y_train passed! That's what makes it unsupervised!)
iso_forest.fit(X_train_scaled)

iso_key = "[1] Original_IF"
models[iso_key] = iso_forest
print("[OK] Done")
print(f"Total models now: {len(models)} (12 supervised + 1 unsupervised)")

# ======================================
# STEP 6: EVALUATE MODELS ON TEST SET (use ONLY X_test_scaled, y_test — NO RESAMPLING!)
# ======================================

# --- Business assumptions for cost analysis (we'll use these later) ---
COST_PER_FALSE_NEGATIVE = 100  # Bank loses $100 avg per missed fraud
COST_PER_FALSE_POSITIVE = 1    # $1 avg cost per false alarm (support calls, customer churn)

print("\n" + "="*110)
print("MODEL EVALUATION COMPARISON TABLE (thresholds chosen on validation data; sorted by Recall desc, then F1 desc)")
print("="*110)
header = f"{'Dataset / Model':<32} {'Threshold':>9} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>8} {'Cost($)':>9}"
print(header)
print("-"*125)

# Store results in a list so we can sort them
results = []
threshold_by_model = {}
pr_auc_by_model = {}

# Choose each supervised model's probability cutoff using validation data only.
# The test set stays untouched until the evaluation loop below.
print("\n--- Choosing supervised-model thresholds on validation data ---")
for model_key, model in models.items():
    if isinstance(model, IsolationForest):
        continue  # Isolation Forest does not produce class probabilities from 0 to 1.

    val_prob = model.predict_proba(X_val_scaled)[:, 1]
    candidate_thresholds = np.arange(0.05, 0.51, 0.05)
    threshold_costs = []
    for threshold in candidate_thresholds:
        val_pred = (val_prob >= threshold).astype(int)
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred).ravel()
        val_cost = val_fn * COST_PER_FALSE_NEGATIVE + val_fp * COST_PER_FALSE_POSITIVE
        threshold_costs.append((val_cost, threshold))

    best_validation_cost, best_threshold = min(threshold_costs)
    threshold_by_model[model_key] = best_threshold
    print(f"{model_key:<32} threshold={best_threshold:.2f}  validation cost=${best_validation_cost:,.0f}")

for model_key, model in models.items():
    # ---- Predict on the TEST SET (the UNALTERED imbalanced one!) ----
    if isinstance(model, IsolationForest):
        # ISOLATION FOREST (UNSUPERVISED):
        # predict() returns: +1 = inlier (normal, Class 0), -1 = outlier (fraud, Class 1)
        y_pred_raw = model.predict(X_test_scaled)
        # Convert +1 -> 0 (normal), -1 -> 1 (fraud)
        y_pred = np.where(y_pred_raw == -1, 1, 0)
        # score_samples() returns: LOWER = more anomalous (fraud)
        # Negate it so HIGHER = more fraudulent (matches roc_auc_score expectation)
        y_prob = -model.score_samples(X_test_scaled)
    else:
        # SUPERVISED MODELS: use the threshold selected earlier on validation data.
        y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of fraud (0.0 to 1.0)
        y_pred = (y_prob >= threshold_by_model[model_key]).astype(int)

    # ---- Compute the 4 confusion matrix numbers ----
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # ravel() unpacks the 2x2 matrix [[tn, fp],[fn, tp]] into 4 integers

    # ---- Compute metrics ----
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    pr_auc_by_model[model_key] = average_precision_score(y_test, y_prob)

    # ---- Business cost calculation ----
    cost = fn * COST_PER_FALSE_NEGATIVE + fp * COST_PER_FALSE_POSITIVE

    # Save for sorting + printing later
    results.append( (model_key, acc, prec, rec, f1, auc, cost, tp, fp, fn, tn) )

# Sort results: best Recall first, break ties with F1, break ties with Cost (lowest best)
results.sort(key=lambda row: (-row[3], -row[4], row[6]))

# Print the sorted table
for model_key, acc, prec, rec, f1, auc, cost, tp, fp, fn, tn in results:
    threshold_display = "IF" if model_key.endswith("_IF") else f"{threshold_by_model[model_key]:.2f}"
    line = (f"{model_key:<32} "
            f"{threshold_display:>9} "
            f"{acc:>9.4f} "
            f"{prec:>10.4f} "
            f"{rec:>8.4f} "
            f"{f1:>7.4f} "
            f"{auc:>8.4f} "
            f"{pr_auc_by_model[model_key]:>8.4f} "
            f"{cost:>9,.0f}")
    print(line)

print("="*110)
print(f"Business cost: ${COST_PER_FALSE_NEGATIVE} per missed fraud (FN), ${COST_PER_FALSE_POSITIVE} per false alarm (FP)")
print("Note: Recall = % of frauds CAUGHT — this is the #1 metric for fraud detection.")
print("="*110)

# ---- Detailed confusion matrix breakdown for the BEST model ----
best_key, best_acc, best_prec, best_rec, best_f1, best_auc, best_cost, tp, fp, fn, tn = results[0]
print(f"\n--- DETAILED BREAKDOWN: Best Model = {best_key} ---")
print(f"  Total transactions in test set: {tn+fp+fn+tp:,}")
print(f"  Actual frauds in test set:      {tp+fn:,}")
print(f"  Fraud transactions CAUGHT (TP): {tp:,}   ({100*tp/(tp+fn):.1f}% of real frauds)")
print(f"  Fraud transactions MISSED (FN): {fn:,}   ({100*fn/(tp+fn):.1f}% of real frauds)")
print(f"  False ALARMS        (FP):       {fp:,}   (legit tx incorrectly flagged)")
print(f"  Legit correctly OK  (TN):       {tn:,}")
print(f"\n  -> Estimated cost for this model on test set:    ${best_cost:,.0f}")
print(f"  -> If we used 'predict always non-fraud' baseline: ${(tp+fn)*COST_PER_FALSE_NEGATIVE:,.0f}")
print(f"  -> SAVINGS vs dumb baseline:                       ${((tp+fn)*COST_PER_FALSE_NEGATIVE - best_cost):,.0f}")

# ---- Feature Importance (from the best RandomForest model) ----
print("\n--- FEATURE IMPORTANCE (from best RandomForest model) ---")
# Find any RF model (we pick the best performing RF from sorted results)
rf_results = [r for r in results if "_RF" in r[0]]
if rf_results:
    best_rf_key = rf_results[0][0]    # Best RF's model_key (first in sorted RF subset)
    best_rf = models[best_rf_key]

    # Pair each column name with its importance score, then sort descending
    importance_pairs = list(zip(X_train.columns, best_rf.feature_importances_))
    importance_pairs.sort(key=lambda pair: pair[1], reverse=True)

    print(f"(Source model: {best_rf_key}) Top 15 most important features for predicting fraud:\n")
    rank = 1
    for feature, score in importance_pairs[:15]:
        bar_len = int(score * 200)  # scale 0-1 score to 0-200 chars for visual bar
        bar = "#" * bar_len
        print(f"  {rank:>2}. {feature:<8}  score={score:.4f}  {bar}")
        rank += 1
    print("\n  Note: Features V1-V28 are PCA-masked so we can't interpret them as 'merchant' / 'location' etc.")
    print("        In a real dataset with raw features, we'd give this list to the fraud investigations team.")

# ---- Save confusion matrices as images (for the TOP 4 best models) ----
print("\n--- Saving confusion matrix heatmaps for Top 4 models (sorted by Recall) ---")
top4 = results[:4]  # First 4 rows from sorted results = best performers
plt.figure(figsize=(16, 10))
plot_index = 1
for model_key, acc, prec, rec, f1, auc, cost, tp, fp, fn, tn in top4:
    cm = [[tn, fp], [fn, tp]]

    plt.subplot(2, 2, plot_index)
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                xticklabels=['Pred Non-Fraud', 'Pred Fraud'],
                yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
    plt.title(f"{model_key}\nRecall={rec:.3f}  Precision={prec:.3f}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plot_index += 1

plt.tight_layout()
plt.savefig(PROJECT_DIR / 'confusion_matrices.png', dpi=100)
plt.close()
print("Saved: confusion_matrices.png (Top 4 models)")

# ---- Final conclusion ----
print("\n" + "="*110)
print("PROJECT COMPLETE! (13 models total: 12 supervised + 1 unsupervised)")
print("="*110)
print("Key takeaways:")
print("  1. Accuracy is meaningless on imbalanced data — a dumb 'all non-fraud' baseline gets 99.83% accuracy.")
print(f"  2. Best Recall = {best_rec*100:.1f}% (caught {best_rec*100:.1f}% of frauds in test set) using {best_key}.")
print(f"  3. That model costs ${best_cost:,.0f} on the test set vs ${(tp+fn)*COST_PER_FALSE_NEGATIVE:,.0f} baseline")
print(f"     -> {100*(((tp+fn)*COST_PER_FALSE_NEGATIVE - best_cost)/((tp+fn)*COST_PER_FALSE_NEGATIVE)):.0f}% cost reduction compared to doing nothing.")
print("  4. Each supervised model's threshold was selected using validation data, then evaluated once on test data.")
print("     Higher recall often means lower precision (more false alarms), so the chosen cost assumptions matter.")

# --- Bagging vs Boosting comparison (interview talking point!) ---
# Find best RF and best XGB for direct comparison
best_rf_row = next((r for r in results if "_RF" in r[0]), None)
best_xgb_row = next((r for r in results if "_XGB" in r[0]), None)
if best_rf_row and best_xgb_row:
    rf_key, rf_acc, rf_prec, rf_rec, rf_f1, rf_auc, rf_cost, *_ = best_rf_row
    xgb_key, xgb_acc, xgb_prec, xgb_rec, xgb_f1, xgb_auc, xgb_cost, *_ = best_xgb_row
    print("\n  5. Bagging (Random Forest) vs Boosting (XGBoost) comparison:")
    print(f"     - Best RF:    {rf_key:<30} Recall={rf_rec*100:5.1f}%  F1={rf_f1:.4f}  Cost=${rf_cost:,.0f}")
    print(f"     - Best XGB:   {xgb_key:<30} Recall={xgb_rec*100:5.1f}%  F1={xgb_f1:.4f}  Cost=${xgb_cost:,.0f}")
    if xgb_rec > rf_rec:
        print(f"     -> XGBoost won by +{100*(xgb_rec-rf_rec):.1f}% recall (boosting often outperforms bagging for structured data)")
    else:
        print(f"     -> Random Forest won by +{100*(rf_rec-xgb_rec):.1f}% recall (rare, but check for overfitting in XGB)")

# --- Supervised vs Unsupervised conclusion (2 sentences as requested) ---
best_supervised = results[0]
best_unsupervised = next((r for r in results if "_IF" in r[0]), None)
if best_unsupervised:
    if_key, if_acc, if_prec, if_rec, if_f1, if_auc, if_cost, *_ = best_unsupervised
    print("\n  6. Supervised vs Unsupervised Paradigm:")
    print(f"     - Best Supervised Recall: {best_supervised[3]*100:.1f}% (requires labeled fraud data for training)")
    print(f"     - Isolation Forest Recall: {if_rec*100:.1f}% (NO labeled fraud data needed at all!)")
print("\n  SUPERVISED MODELS were stronger here because they learn from labelled fraud examples.")
print("  Isolation Forest can still help when labelled fraud data is unavailable, but its test metrics should guide deployment.")
print("="*110)












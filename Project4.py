#Stress Level Prediction from Multimodal Wearable & Smartphone Data 
#Anne McCullagh, Brady Galligan, Luke Mele, Thomas Rua

import os
import glob
import pandas as pd

DATASET_PATH = "Loneliness_Dataset_Nov10" #dataset path

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all participant folders
participant_dirs = sorted(glob.glob(os.path.join(DATASET_PATH, "Participant_*")))
participant_ids = [os.path.basename(d) for d in participant_dirs]
print(f"Found {len(participant_ids)} participant folders")
print(f"Participants: {participant_ids[:5]}... (showing first 5)")


# =============================================================================
# TASK 1: DATASET UNDERSTANDING
# =============================================================================
print("\n" + "="*70)
print("TASK 1: DATASET UNDERSTANDING")
print("="*70)
print("""
PURPOSE: The dataset captures multimodal daily data on physiological,
behavioral, and psychological measures from first-generation immigrants
in Finland to study loneliness and mental well-being.
""")


# =============================================================================
# TASK 2: DATA EXPLORATION
# =============================================================================
print("\n" + "="*70)
print("TASK 2: DATA EXPLORATION")
print("="*70)

# --- Explore what files each participant has ---
def explore_participant(p_dir):
    """Explore a single participant's data availability."""
    pid = os.path.basename(p_dir)
    info = {"participant": pid}
    
    # Check Aware folder
    aware_dir = os.path.join(p_dir, "Aware")
    if os.path.exists(aware_dir):
        aware_files = [f for f in os.listdir(aware_dir) if f.endswith('.csv')]
        info["aware_files"] = aware_files
        info["has_aware"] = len(aware_files) > 0
    else:
        info["aware_files"] = []
        info["has_aware"] = False
    
    # Check Oura folder
    oura_dir = os.path.join(p_dir, "Oura")
    if os.path.exists(oura_dir):
        oura_files = [f for f in os.listdir(oura_dir) if f.endswith('.csv')]
        info["has_oura"] = len(oura_files) > 0
        info["oura_file"] = oura_files[0] if oura_files else None
    else:
        info["has_oura"] = False
        info["oura_file"] = None
    
    # Check Surveys folder
    survey_dir = os.path.join(p_dir, "Surveys")
    if os.path.exists(survey_dir):
        survey_files = [f for f in os.listdir(survey_dir) if f.endswith('.csv')]
        info["survey_files"] = survey_files
        info["has_surveys"] = len(survey_files) > 0
        info["has_pss_weekly"] = any("stress every week" in f.lower() or 
                                     "perceived stress every" in f.lower() or
                                     "stress_every_week" in f.lower()
                                     for f in survey_files)
        info["has_ema"] = any("ema" in f.lower() for f in survey_files)
    else:
        info["survey_files"] = []
        info["has_surveys"] = False
        info["has_pss_weekly"] = False
        info["has_ema"] = False
    
    # Check Watch folder
    watch_dir = os.path.join(p_dir, "Watch")
    if os.path.exists(watch_dir):
        watch_files = [f for f in os.listdir(watch_dir) if f.endswith('.csv')]
        info["has_watch"] = len(watch_files) > 0
        info["num_watch_files"] = len(watch_files)
    else:
        info["has_watch"] = False
        info["num_watch_files"] = 0
    
    return info

# Explore all participants
exploration_data = []
for p_dir in participant_dirs:
    exploration_data.append(explore_participant(p_dir))

exploration_df = pd.DataFrame(exploration_data)

print(f"\nTotal participants found: {len(exploration_df)}")
print(f"Participants with Aware data: {exploration_df['has_aware'].sum()}")
print(f"Participants with Oura data: {exploration_df['has_oura'].sum()}")
print(f"Participants with Watch data: {exploration_df['has_watch'].sum()}")
print(f"Participants with Survey data: {exploration_df['has_surveys'].sum()}")
print(f"Participants with weekly PSS: {exploration_df['has_pss_weekly'].sum()}")

# Show survey files for one participant to understand naming
if not exploration_df.empty:
    sample_p = exploration_data[0]
    print(f"\nSample survey files for {sample_p['participant']}:")
    for f in sorted(sample_p.get('survey_files', [])):
        print(f"  - {f}")
# =============================================================================
# TASK 3: DATA PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("TASK 3: DATA PREPROCESSING")
print("="*70)

import numpy as np

def find_csv_file(folder, keywords=None):
    """Return first CSV file in folder matching any keyword."""
    if not os.path.exists(folder):
        return None
    
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if keywords is None:
        return os.path.join(folder, files[0]) if files else None
    
    for f in files:
        fname = f.lower()
        if any(k.lower() in fname for k in keywords):
            return os.path.join(folder, f)
    return None

def safe_read_csv(path):
    """Read CSV safely."""
    if path is None or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return None

def find_datetime_column(df):
    """Try to detect datetime column."""
    if df is None:
        return None
    
    candidates = [
        "timestamp", "datetime", "date", "local_date", "time",
        "start_time", "end_time", "created_at"
    ]
    
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None

def convert_to_date(df):
    """Convert datetime column to date."""
    if df is None or df.empty:
        return None
    
    dt_col = find_datetime_column(df)
    if dt_col is None:
        return None
    
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df["date"] = df[dt_col].dt.date
        return df
    except:
        return None

def numeric_columns(df):
    """Return numeric columns only."""
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def daily_aggregate(df, prefix):
    """
    Aggregate numeric columns by day.
    Creates mean values for numeric columns.
    """
    if df is None or df.empty:
        return None
    
    df = convert_to_date(df)
    if df is None or "date" not in df.columns:
        return None
    
    num_cols = numeric_columns(df)
    if not num_cols:
        return None
    
    grouped = df.groupby("date")[num_cols].mean().reset_index()
    grouped = grouped.rename(columns={col: f"{prefix}_{col}" for col in num_cols})
    return grouped

def event_count_by_day(df, prefix):
    """
    Count number of rows/events per day for event-style data
    such as notifications, calls, messages, screen events.
    """
    if df is None or df.empty:
        return None
    
    df = convert_to_date(df)
    if df is None or "date" not in df.columns:
        return None
    
    counts = df.groupby("date").size().reset_index(name=f"{prefix}_count")
    return counts

all_daily_features = []

for p_dir in participant_dirs:
    pid = os.path.basename(p_dir)
    print(f"Processing {pid}...")
    
    participant_daily = None
    
    # ---------------------------
    # AWARE DATA
    # ---------------------------
    aware_dir = os.path.join(p_dir, "Aware")
    
    notif_file = find_csv_file(aware_dir, ["notification"])
    call_file = find_csv_file(aware_dir, ["call"])
    message_file = find_csv_file(aware_dir, ["message", "sms"])
    screen_file = find_csv_file(aware_dir, ["screen"])
    battery_file = find_csv_file(aware_dir, ["battery"])
    
    notif_df = safe_read_csv(notif_file)
    call_df = safe_read_csv(call_file)
    message_df = safe_read_csv(message_file)
    screen_df = safe_read_csv(screen_file)
    battery_df = safe_read_csv(battery_file)
    
    notif_daily = event_count_by_day(notif_df, "notif")
    call_daily = event_count_by_day(call_df, "call")
    message_daily = event_count_by_day(message_df, "message")
    screen_daily = event_count_by_day(screen_df, "screen")
    battery_daily = daily_aggregate(battery_df, "battery")
    
    aware_parts = [notif_daily, call_daily, message_daily, screen_daily, battery_daily]
    aware_parts = [x for x in aware_parts if x is not None]
    
    if aware_parts:
        aware_merged = aware_parts[0]
        for part in aware_parts[1:]:
            aware_merged = pd.merge(aware_merged, part, on="date", how="outer")
        participant_daily = aware_merged
    
    # ---------------------------
    # OURA DATA
    # ---------------------------
    oura_dir = os.path.join(p_dir, "Oura")
    oura_file = find_csv_file(oura_dir)
    oura_df = safe_read_csv(oura_file)
    
    oura_daily = daily_aggregate(oura_df, "oura")
    if oura_daily is not None:
        if participant_daily is None:
            participant_daily = oura_daily
        else:
            participant_daily = pd.merge(participant_daily, oura_daily, on="date", how="outer")
    
    # ---------------------------
    # WATCH DATA
    # ---------------------------
    # ---------------------------
# WATCH DATA
# ---------------------------
watch_dir = os.path.join(p_dir, "Watch")
if os.path.exists(watch_dir):
    watch_files = [
        os.path.join(watch_dir, f)
        for f in os.listdir(watch_dir)
        if f.endswith(".csv")
    ]

    watch_dfs = []

    for wf in watch_files[:5]:   # keep your current limit for now
        wdf = safe_read_csv(wf)
        if wdf is not None and not wdf.empty:
            watch_dfs.append(wdf)

    if watch_dfs:
        combined_watch_df = pd.concat(watch_dfs, ignore_index=True)
        watch_daily = daily_aggregate(combined_watch_df, "watch")

        if watch_daily is not None:
            if participant_daily is None:
                participant_daily = watch_daily
            else:
                participant_daily = pd.merge(
                    participant_daily,
                    watch_daily,
                    on="date",
                    how="outer"
                )
    
    # ---------------------------
    # ADD PARTICIPANT ID
    # ---------------------------
    if participant_daily is not None and not participant_daily.empty:
        participant_daily["participant"] = pid
        all_daily_features.append(participant_daily)

# Combine all participants
if all_daily_features:
    daily_features_df = pd.concat(all_daily_features, ignore_index=True)
else:
    daily_features_df = pd.DataFrame()

print(f"\nCombined daily feature table shape: {daily_features_df.shape}")
print("Columns:")
print(daily_features_df.columns.tolist()[:20])

# ---------------------------
# HANDLE MISSING VALUES
# ---------------------------
if not daily_features_df.empty:
    numeric_cols = daily_features_df.select_dtypes(include=[np.number]).columns
    
    # Drop columns with too much missing data (>50%)
    missing_ratio = daily_features_df[numeric_cols].isnull().mean()
    keep_cols = missing_ratio[missing_ratio <= 0.5].index.tolist()
    
    base_cols = ["participant", "date"]
    daily_features_df = daily_features_df[base_cols + keep_cols]
    
    # Median imputation for remaining numeric missing values
    for col in keep_cols:
        daily_features_df[col] = daily_features_df[col].fillna(daily_features_df[col].median())
    
    # Remove duplicates
    daily_features_df = daily_features_df.drop_duplicates(subset=["participant", "date"])

print(f"\nDaily feature table after cleaning: {daily_features_df.shape}")

# Save preprocessed features
daily_features_path = os.path.join(OUTPUT_DIR, "daily_features.csv")
daily_features_df.to_csv(daily_features_path, index=False)
print(f"Saved daily features to: {daily_features_path}")
# =============================================================================
# TASK 4: DEFINE A PREDICTION TASK
# =============================================================================
print("\n" + "="*70)
print("TASK 4: DEFINE A PREDICTION TASK")
print("="*70)

def find_stress_survey_file(survey_dir):
    """Find a likely stress-related survey CSV."""
    if not os.path.exists(survey_dir):
        return None
    
    survey_files = [f for f in os.listdir(survey_dir) if f.endswith(".csv")]
    
    for f in survey_files:
        name = f.lower()
        if ("stress" in name) or ("pss" in name) or ("perceived stress" in name):
            return os.path.join(survey_dir, f)
    return None

def extract_stress_labels(p_dir):
    """
    Extract stress labels from survey file.
    This tries to:
    1. find a stress-related survey file
    2. identify a score column
    3. convert date
    """
    pid = os.path.basename(p_dir)
    survey_dir = os.path.join(p_dir, "Surveys")
    stress_file = find_stress_survey_file(survey_dir)
    
    if stress_file is None:
        return None
    
    df = safe_read_csv(stress_file)
    if df is None or df.empty:
        return None
    
    df = convert_to_date(df)
    if df is None:
        return None
    
    # Try to locate a stress score column
    possible_score_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if ("stress" in col_lower) or ("pss" in col_lower) or ("score" in col_lower):
            possible_score_cols.append(col)
    
    # Keep only numeric candidate columns
    score_col = None
    for col in possible_score_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            score_col = col
            break
    
    if score_col is None:
        # fallback: first numeric column not date
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            score_col = num_cols[0]
        else:
            return None
    
    out = df[["date", score_col]].copy()
    out["participant"] = pid
    out = out.rename(columns={score_col: "stress_score"})
    out = out.dropna(subset=["stress_score"])
    
    return out

# Collect stress labels from all participants
stress_label_list = []

for p_dir in participant_dirs:
    stress_df = extract_stress_labels(p_dir)
    if stress_df is not None and not stress_df.empty:
        stress_label_list.append(stress_df)

if stress_label_list:
    stress_labels_df = pd.concat(stress_label_list, ignore_index=True)
else:
    stress_labels_df = pd.DataFrame(columns=["participant", "date", "stress_score"])

print(f"Stress label rows found: {len(stress_labels_df)}")

if not stress_labels_df.empty:
    print("\nSample stress labels:")
    print(stress_labels_df.head())

# ----------------------------------------
# CREATE BINARY TARGET: HIGH vs LOW STRESS
# ----------------------------------------
if not stress_labels_df.empty:
    median_stress = stress_labels_df["stress_score"].median()
    stress_labels_df["stress_label"] = (stress_labels_df["stress_score"] >= median_stress).astype(int)
    
    print(f"\nMedian stress score used as cutoff: {median_stress}")
    print("Label distribution:")
    print(stress_labels_df["stress_label"].value_counts())

# ----------------------------------------
# MERGE FEATURES WITH LABELS
# ----------------------------------------
if not daily_features_df.empty and not stress_labels_df.empty:
    modeling_df = pd.merge(
        daily_features_df,
        stress_labels_df,
        on=["participant", "date"],
        how="inner"
    )
else:
    modeling_df = pd.DataFrame()

print(f"\nFinal modeling dataset shape: {modeling_df.shape}")

if not modeling_df.empty:
    print("\nTarget variable distribution:")
    print(modeling_df["stress_label"].value_counts())

# Save modeling dataset
modeling_path = os.path.join(OUTPUT_DIR, "stress_modeling_dataset.csv")
modeling_df.to_csv(modeling_path, index=False)
print(f"Saved modeling dataset to: {modeling_path}")


# =============================================================================
# TASK 5: APPLY MACHINE LEARNING MODELS
# =============================================================================
print("\n" + "="*70)
print("TASK 5: APPLY MACHINE LEARNING MODELS")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# XGBoost import
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    print("XGBoost is not installed. Run: pip install xgboost")
    xgb_available = False

if modeling_df.empty:
    print("Modeling dataset is empty. Cannot train models.")
else:
    # ---------------------------
    # PREPARE X AND y
    # ---------------------------
    drop_cols = ["participant", "date", "stress_score", "stress_label"]
    feature_cols = [c for c in modeling_df.columns if c not in drop_cols]

    X = modeling_df[feature_cols].copy()
    y = modeling_df["stress_label"].copy()

    print(f"Number of rows in modeling dataset: {len(modeling_df)}")
    print(f"Number of features: {len(feature_cols)}")

    # Optional: remove zero-variance columns
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]
        print(f"Removed {len(constant_cols)} constant feature columns")

    # ---------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # DEFINE MODELS
    # ---------------------------
    models = {
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ))
        ]),

        "Neural Network": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42
            ))
        ])
    }

    if xgb_available:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])

    # ---------------------------
    # TRAIN AND EVALUATE
    # ---------------------------
    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # probability or decision score for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if y_score is not None:
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = float("nan")

        results.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        })

        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    print("\nModel Comparison:")
    print(results_df)

    results_path = os.path.join(OUTPUT_DIR, "task5_model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved Task 5 results to: {results_path}")

    # ---------------------------
    # FEATURE IMPORTANCE FOR TREE MODELS
    # ---------------------------
    print("\nTop Features:")
    for model_name, model in models.items():
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=X.columns)
            top_feats = importances.sort_values(ascending=False).head(10)
            print(f"\n{model_name} top 10 features:")
            print(top_feats)

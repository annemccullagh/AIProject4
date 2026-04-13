#Stress Level Prediction from Multimodal Wearable & Smartphone Data

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


import pandas as pd
import numpy as np
import random

# Set a seed for reproducibility
np.random.seed(42)

# Number of rows
n = 2000

# --- 1. Feature Generation ---
patch_ids = [f"P{1000 + i}" for i in range(n)]
os_types = np.random.choice(["RHEL", "Windows", "Ubuntu"], size=n, p=[0.5, 0.3, 0.2])
patch_types = np.random.choice(["Security", "Bugfix", "Feature", "Kernel"], size=n, p=[0.4, 0.3, 0.2, 0.1])
patch_severity = np.random.choice(["Low", "Medium", "High", "Critical"], size=n, p=[0.3, 0.4, 0.2, 0.1])
systems_affected = np.random.randint(5, 100, size=n)
server_usage = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.35, 0.45, 0.2])
prev_incidents = np.random.poisson(lam=0.5, size=n).clip(0, 5) 
patch_size_mb = np.random.randint(50, 500, size=n)
applied_during_peak = np.random.choice(["Yes", "No"], size=n, p=[0.2, 0.8])
pre_health_score = np.random.normal(loc=85, scale=10, size=n).clip(40, 100)


# --- 2. Calculate Pre-Patch Risk Score (The new, clean target driver) ---
# This complex score uses non-linear interactions to define high risk.
risk_score = (
    # Base risk from critical patch types
    np.where((patch_types == "Kernel") | (patch_types == "Feature"), 3, 0)
    # Risk multiplier based on Severity
    + np.where(patch_severity == "Critical", 5, 
               np.where(patch_severity == "High", 3, 0))
    # Non-linear interaction: Poor Health * Previous Incidents (High Risk Factor)
    + ((100 - pre_health_score) / 10) * (prev_incidents + 1)
    # Risk from applying during high load
    + np.where(applied_during_peak == "Yes", 2, 0)
    + np.where(server_usage == "High", 1, 0)
    # Risk from size (normalized)
    + (patch_size_mb / 200)
)

# --- 3. Result Classification with Noise (CRITICAL FIX) ---
# Define baseline result based *only* on the complex Pre-Patch Risk Score.
# We set the top 30% risk_score as 'Risky'
risk_threshold = np.percentile(risk_score, 70)
baseline_result = np.where(risk_score >= risk_threshold, "Risky", "Safe") 

# Introduce structured noise (10% chance to flip the baseline result)
result = np.copy(baseline_result)
for i in range(n):
    if np.random.rand() < 0.10: 
        result[i] = "Safe" if result[i] == "Risky" else "Risky"


# --- 4. Post-patch Impact Simulation (LEAKAGE FEATURES - RETAINED FOR REALISM) ---
# These are still generated, but they no longer define the 'Result'.
# Downtime/Post_Health are now a noisy *consequence* of the Risk Score.
impact_factor = -1 * (risk_score * np.random.uniform(0.5, 1.5, size=n) + np.random.normal(0, 2, size=n))
post_health_score = (pre_health_score + impact_factor).clip(40, 100)
health_drop = np.abs(pre_health_score - post_health_score) 
downtime_mins = (health_drop * np.random.uniform(0.5, 2.0, size=n)).round(1).clip(0, 45)


# --- 5. Assemble DataFrame ---
df = pd.DataFrame({
    "Patch_ID": patch_ids,
    "OS_Type": os_types,
    "Patch_Type": patch_types,
    "Patch_Severity": patch_severity,
    "Systems_Affected": systems_affected,
    "Server_Usage": server_usage,
    "Prev_Incidents": prev_incidents,
    "Patch_Size_MB": patch_size_mb,
    "Applied_During_Peak": applied_during_peak,
    "Pre_Health_Score": pre_health_score.round(1),
    "Post_Health_Score": post_health_score.round(1),
    "Downtime_Mins": downtime_mins,
    "Result": result
})

# Save to CSV
df.to_csv("patch_data_revised.csv", index=False)

print("✅ FINAL Revised Dataset created: patch_data_revised.csv with 2000 rows. The Result is now strongly tied to Pre-Patch features.")
df.head(10)


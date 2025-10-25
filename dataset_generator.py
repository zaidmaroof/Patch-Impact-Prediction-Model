import pandas as pd
import numpy as np
import random

# Set a seed for reproducibility
np.random.seed(42)

# Number of rows
n = 2000

# --- 1. Original Feature Generation ---
patch_ids = [f"P{1000 + i}" for i in range(n)]
os_types = np.random.choice(["RHEL", "Windows", "Ubuntu"], size=n, p=[0.5, 0.3, 0.2])
patch_types = np.random.choice(["Security", "Bugfix", "Feature", "Kernel"], size=n, p=[0.4, 0.3, 0.2, 0.1])
patch_severity = np.random.choice(["Low", "Medium", "High", "Critical"], size=n, p=[0.3, 0.4, 0.2, 0.1])
systems_affected = np.random.randint(5, 100, size=n)
pre_health_score = np.random.normal(loc=85, scale=10, size=n).clip(40, 100)

# --- 2. New Predictive Features Added (Pre-Patch) ---
server_usage = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.35, 0.45, 0.2])
# Previous Incidents: Use Poisson distribution to get mostly 0s and 1s, but sometimes more
prev_incidents = np.random.poisson(lam=0.5, size=n).clip(0, 5) 
patch_size_mb = np.random.randint(50, 500, size=n)
applied_during_peak = np.random.choice(["Yes", "No"], size=n, p=[0.2, 0.8])

# --- 3. Post-patch Impact Simulation (RISK MODELLING) ---
impact_factor = []
for i in range(n):
    sev = patch_severity[i]
    typ = patch_types[i]
    os_ = os_types[i]
    
    # Base instability with a small mean to allow for improvement in some cases
    base = np.random.normal(1, 5) 

    # Impact from Severity and Type (High impact features)
    if sev == "Critical": base -= np.random.uniform(10, 25)
    elif sev == "High": base -= np.random.uniform(5, 15)
    elif sev == "Medium": base -= np.random.uniform(0, 8)
    else: base -= np.random.uniform(-3, 3)
    
    if typ == "Kernel": base -= np.random.uniform(5, 15)
    if os_ == "Windows": base -= np.random.uniform(-2, 5)

    # Impact from NEW Pre-patch features (Model Predictors)
    if server_usage[i] == "High": base -= np.random.uniform(3, 8)
    elif server_usage[i] == "Medium": base -= np.random.uniform(0, 3)
    
    base -= prev_incidents[i] * np.random.uniform(1, 4) # Higher incidents means higher base impact
    
    if applied_during_peak[i] == "Yes": base -= np.random.uniform(2, 5)
    
    # Impact from Patch_Size_MB: Larger size means slightly larger negative impact
    base -= (patch_size_mb[i] / 500) * np.random.uniform(0, 5)

    impact_factor.append(base)

# Post-patch health score (Post-Patch Leaking Feature 1)
post_health_score = (pre_health_score + np.array(impact_factor)).clip(0, 100)

# Downtime minutes (Post-Patch Leaking Feature 2)
# Downtime is proportional to the health drop
health_drop = np.abs(pre_health_score - post_health_score) 
downtime_mins = health_drop * np.random.uniform(0.5, 1.5, size=n)
downtime_mins = downtime_mins.round(1)

# --- 4. Result Classification BASED ON PRE-PATCH RISK ---

# Calculate a simple Pre-Patch Risk Score (higher score = higher inherent risk)
risk_score = (
    # Patch Type: Kernel/Feature are higher risk
    np.where((patch_types == "Kernel") | (patch_types == "Feature"), 1.5, 0)
    # Patch Severity: Higher severity is higher risk
    + np.where(patch_severity == "Critical", 2.0, 
               np.where(patch_severity == "High", 1.0, 0))
    # Previous Incidents: Direct risk indicator
    + prev_incidents * 1.5
    # Pre-Health Score: Lower health is higher risk (Inverse relationship)
    + (100 - pre_health_score) / 20 
    # Applied During Peak: Higher risk
    + np.where(applied_during_peak == "Yes", 1.0, 0)
    # Server Usage: High usage is higher risk
    + np.where(server_usage == "High", 1.5, 0)
)

# Use a threshold on the risk score to define the baseline result
baseline_result = np.where(risk_score > np.percentile(risk_score, 70), "Risky", "Safe") 
# This sets the top 30% riskiest patches based on Pre-Patch features as "Risky"

# Introduce Noise (Keep this to simulate real-world errors)
result = np.copy(baseline_result)
for i in range(n):
    # Noise: 15% chance to flip the baseline result
    if np.random.rand() < 0.15: 
        result[i] = "Safe" if result[i] == "Risky" else "Risky"

# Assemble DataFrame
df = pd.DataFrame({
    "Patch_ID": patch_ids,
    "OS_Type": os_types,
    "Patch_Type": patch_types,
    "Patch_Severity": patch_severity,
    "Systems_Affected": systems_affected,
    "Server_Usage": server_usage,          # NEW
    "Prev_Incidents": prev_incidents,      # NEW
    "Patch_Size_MB": patch_size_mb,        # NEW
    "Applied_During_Peak": applied_during_peak, # NEW
    "Pre_Health_Score": pre_health_score.round(1),
    "Post_Health_Score": post_health_score.round(1),
    "Downtime_Mins": downtime_mins,
    "Result": result
})

# Save to CSV
df.to_csv("patch_data_revised.csv", index=False)

print("✅ Revised Dataset created: patch_data_revised.csv with 2000 rows.")
df.head(10)

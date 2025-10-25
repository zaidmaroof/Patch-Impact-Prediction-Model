# Patch Impact Prediction Dataset

This dataset simulates real-world **infrastructure patch management data** across enterprise servers.  
The goal is to **predict whether a patch will be *Safe* or *Risky*** before applying it — helping IT teams minimize post-patch incidents and downtime.

---

## Dataset Overview

| Item | Description |
|------|--------------|
| **File Name** | `patch_data_revised.csv` |
| **Records** | 2,000 |
| **Total Features** | 13 (including target) |
| **Prediction Goal** | Predict `Result` (Safe / Risky) based on pre-patch data |
| **Data Type** | Mixed — categorical, numerical, boolean |
| **Leakage** | No post-patch leakage (target derived from pre-patch features only) |

---

## Feature Description

| Feature | Type | Example | Description |
|----------|------|----------|--------------|
| **Patch_ID** | String / ID | `P1024` | Unique identifier for each patch record. Not predictive. |
| **OS_Type** | Categorical | `RHEL`, `Windows`, `Ubuntu` | Operating system where the patch is applied. |
| **Patch_Type** | Categorical | `Security`, `Bugfix`, `Feature`, `Kernel` | Type of patch. Kernel and Feature updates carry higher risk. |
| **Patch_Severity** | Categorical | `Low`, `Medium`, `High`, `Critical` | Severity level or importance of the patch. |
| **Systems_Affected** | Numeric (int) | `72` | Number of systems affected by the patch rollout. |
| **Server_Usage** | Categorical | `Low`, `Medium`, `High` | Typical workload level of the server. High usage = higher risk. |
| **Prev_Incidents** | Numeric (int) | `2` | Number of past patch-related incidents on this system. |
| **Patch_Size_MB** | Numeric (int) | `340` | Size of the patch file in MB. Larger patches can be riskier. |
| **Applied_During_Peak** | Categorical (Yes/No) | `Yes` | Whether the patch was applied during business peak hours. |
| **Pre_Health_Score** | Numeric (float, 0–100) | `83.4` | Overall system health before patching (based on CPU, disk, service metrics). |
| **Post_Health_Score** | Numeric (float, 0–100) | `78.9` | Health score after patching — *for analysis only*, not for prediction. |
| **Downtime_Mins** | Numeric (float) | `12.6` | Minutes of downtime observed — *for analysis only*, not for prediction. |
| **Result** | Categorical (Target) | `Safe`, `Risky` | Label determined from pre-patch risk indicators (top 30% risky patches + 15% noise). |

---

## Target Label Logic

`Result` is classified as **Risky** if the patch has a high combined risk score based on:

- Patch severity (`High`, `Critical`)
- Patch type (`Kernel`, `Feature`)
- Number of previous incidents
- Low pre-patch health score
- High server usage
- Applied during peak hours
- Large patch size

The top 30% riskiest patches are marked as **Risky**, with 15% label noise added to mimic real-world unpredictability.

---

## Features to Exclude During Training

| Feature | Reason |
|----------|---------|
| `Patch_ID` | Identifier only |
| `Post_Health_Score` | Post-patch metric (data leakage) |
| `Downtime_Mins` | Post-patch metric (data leakage) |

---

## Modeling Objective

Train an ML model (e.g., **Random Forest**, **Logistic Regression**, **XGBoost**) to predict whether a patch is *Safe* or *Risky* **before it is applied**, using only pre-patch features.

Evaluate model performance with:

- Accuracy, Precision, Recall, F1-Score  
- Feature importance ranking  
- SHAP or permutation-based interpretability

---

## Potential Use Cases

- Automate **patch risk assessment** in CI/CD pipelines  
- Integrate predictions into **ServiceNow change workflows**  
- Reduce post-patch **incident frequency and downtime**  
- Build an **AI-powered patch recommendation system**

---

## Example Applications

- **Patch Impact Prediction Model**  
  Predict patch risk using historical deployment data.

- **Downtime Estimator**  
  Use regression models to estimate expected downtime duration.

- **Root Cause Correlation**  
  Identify top contributing features for risky patch behavior.

---

## Notes

- Dataset is **synthetic** but modeled after real enterprise patching behavior.  
- Generated with a fixed random seed (`np.random.seed(42)`) for reproducibility.  
- Free for experimentation and educational use.

---
## Real-world Data Mapping

In this synthetic dataset, features such as Pre_Health_Score, Post_Health_Score, and Server_Usage are randomly simulated. However, in real production environments, these values can be derived from system monitoring tools and observability platforms.

Pre_Health_Score can be computed as a composite metric combining system KPIs like CPU utilization, memory consumption, disk I/O performance, network latency, and service uptime before patching.

Post_Health_Score represents the same metric captured after patch deployment, indicating post-change system stability.

Server_Usage can be estimated from long-term workload statistics, such as average CPU utilization, IOPS (input/output operations per second), or active user sessions, typically collected from infrastructure monitoring dashboards or automation frameworks (e.g., Prometheus, Grafana, Zabbix, Splunk, or ServiceNow Discovery).

A possible computation formula for Pre_Health_Score could look like this:
```
Pre_Health_Score = 100 - (
    0.4 * avg_cpu_utilization +
    0.3 * avg_memory_utilization +
    0.2 * disk_latency_score +
    0.1 * network_error_rate
)
```
Where:

Each component (CPU, memory, disk, network) is normalized on a scale of 0–100.

The weights (0.4, 0.3, etc.) reflect their relative importance in overall system stability.

Higher values of Pre_Health_Score indicate better pre-patch system health.

In a real-world ML pipeline, these metrics would be automatically collected from APIs or monitoring logs and aggregated into a unified health score. This approach ensures that predictions are grounded in live operational data rather than static or random values.

---
### Author : https://github.com/zaidmaroof
**Patch Impact Prediction Dataset (v2)**  
Created for hands-on machine learning experimentation on **Linux Mint** or other Python environments.

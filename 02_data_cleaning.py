import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)
print(" STEP 2: Data Cleaning & Feature Engineering")


df = pd.read_csv("data/raw/district_welfare_data.csv")
print(f"\n Loaded {len(df)} districts, {df.shape[1]} columns")

print("\n[1] Null Check:")
nulls = df.isnull().sum()
print(f"    Total nulls: {nulls.sum()} — {'None' if nulls.sum() == 0 else 'Fix needed'}")

print("\n[2] Range Validation:")

RANGE_CHECKS = {
    "pmkisan_uptake_rate": (0.0, 1.0),
    "ab_uptake_rate": (0.0, 1.0),
    "literacy_rate": (0.0, 1.0),
    "rural_pct": (0.0, 1.0),
    "bpl_pct": (0.0, 1.0),
    "sc_st_pct": (0.0, 1.0),
    "development_index": (0.0, 1.0),
}

for col, (lo, hi) in RANGE_CHECKS.items():
    violations = ((df[col] < lo) | (df[col] > hi)).sum()
    status = "yes" if violations == 0 else f" {violations} violations"
    print(f"    {col:<35} {status}")

print("\n[3] Outlier Detection (IQR):")

numeric_cols = [
    "population_lakhs",
    "health_facilities_per_lakh",
    "avg_landholding_ha",
]

outlier_log = []

for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    n_out = ((df[col] < lo) | (df[col] > hi)).sum()

    outlier_log.append({
        "column": col,
        "Q1": Q1,
        "Q3": Q3,
        "lower_fence": lo,
        "upper_fence": hi,
        "outliers": n_out,
    })

    print(
        f"    {col:<38} {n_out} outliers "
        f"(IQR fences: [{lo:.2f}, {hi:.2f}])"
    )

for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

print("    → Outliers Winsorized at IQR fences")

print("\n[4] Feature Engineering:")

def gap_tier(rate):
    if rate < 0.40:
        return "Critical"
    elif rate < 0.60:
        return "High"
    elif rate < 0.75:
        return "Moderate"
    else:
        return "Low"


df["pmkisan_gap_tier"] = df["pmkisan_uptake_rate"].apply(gap_tier)
df["ab_gap_tier"] = df["ab_uptake_rate"].apply(gap_tier)

df["pmkisan_gap_rate"] = 1 - df["pmkisan_uptake_rate"]
df["ab_gap_rate"] = 1 - df["ab_uptake_rate"]

df["priority_score"] = (
    df["combined_gap_score"] * 0.40 +
    (
        (df["pmkisan_gap_count"] + df["ab_gap_count"]) /
        (df["pmkisan_gap_count"] + df["ab_gap_count"]).max()
    ) * 0.30 +
    df["bpl_pct"] * 0.30
)

df["priority_score"] = df["priority_score"].round(4)

df["digital_access_index"] = (
    df["internet_penetration"] * 0.60 +
    df["literacy_rate"] * 0.40
).round(4)

df["gender_literacy_gap"] = (
    df["literacy_rate"] - df["female_literacy_rate"]
).round(4)

df["log_population"] = np.log1p(df["population_lakhs"]).round(4)

state_agg = df.groupby("state").agg(
    state_avg_pmkisan_uptake=("pmkisan_uptake_rate", "mean"),
    state_avg_ab_uptake=("ab_uptake_rate", "mean"),
    state_total_pmkisan_gap=("pmkisan_gap_count", "sum"),
    state_total_ab_gap=("ab_gap_count", "sum"),
).reset_index()

df = df.merge(state_agg, on="state", how="left")

print(
    "    Added columns: pmkisan_gap_tier, ab_gap_tier, "
    "priority_score, digital_access_index, "
    "gender_literacy_gap, log_population, state-level aggregates"
)

print("\n[5] Key Statistics:")

key_stats = df[
    [
        "pmkisan_uptake_rate",
        "ab_uptake_rate",
        "combined_gap_score",
        "literacy_rate",
        "bpl_pct",
        "rural_pct",
    ]
].describe().round(3)

print(key_stats.to_string())

print("\n[6] Gap Tier Distribution (PM-KISAN):")
print(df["pmkisan_gap_tier"].value_counts().to_string())

print("\n[7] Gap Tier Distribution (Ayushman Bharat):")
print(df["ab_gap_tier"].value_counts().to_string())

df.to_csv(
    "data/processed/district_data_clean.csv",
    index=False
)

pd.DataFrame(outlier_log).to_csv(
    "data/processed/outlier_log.csv",
    index=False
)

print("\nCleaned dataset saved: data/processed/district_data_clean.csv")
print(f"   Shape: {df.shape}")


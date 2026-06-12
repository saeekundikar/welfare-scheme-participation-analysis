import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

import warnings

warnings.filterwarnings("ignore")


os.makedirs("outputs/maps", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


sns.set_theme(
    style="whitegrid",
    font_scale=1.1
)


plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


COLORS = {
    "High Coverage": "#06D6A0",
    "Medium Coverage": "#FFD166",
    "Low Coverage": "#F8961E",
    "Critical Priority": "#EF476F"
}


print("=" * 60)
print("STEP 6: Intervention Priority Mapping")
print("=" * 60)


df = pd.read_csv(
    "data/processed/district_data_clean.csv"
)


print(
    f"SUCCESS: Loaded {len(df)} districts"
)


print("\nCreating state-level intervention summary")


state_summary = (
    df.groupby("state")
    .agg(
        districts=("district_id", "count"),
        average_priority_score=("priority_score", "mean"),
        pmkisan_uptake=("pmkisan_uptake_rate", "mean"),
        ayushman_uptake=("ab_uptake_rate", "mean"),
        critical_districts=(
            "district_segment",
            lambda x: (x == "Critical Priority").sum()
        ),
        high_risk_districts=(
            "district_segment",
            lambda x: x.isin(
                [
                    "Critical Priority",
                    "Low Coverage"
                ]
            ).sum()
        )
    )
    .round(3)
)


state_summary = (
    state_summary
    .sort_values(
        "average_priority_score",
        ascending=False
    )
)


print("\nTop 10 States Requiring Immediate Intervention")

print(
    state_summary
    .head(10)
    .to_string()
)


fig, ax = plt.subplots(
    figsize=(12, 8)
)


top_states = state_summary.head(15)


bars = ax.barh(
    top_states.index[::-1],
    top_states["average_priority_score"][::-1],
    color=COLORS["Critical Priority"],
    alpha=0.8
)


ax.set_title(
    "States with Highest Welfare Intervention Priority",
    fontsize=14,
    fontweight="bold"
)


ax.set_xlabel(
    "Average Priority Score"
)


for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center"
    )


plt.savefig(
    "outputs/maps/map01_high_priority_states.png"
)


plt.close()


print(
    "SUCCESS: State priority ranking map saved"
)

print("\nCreating district segment distribution analysis")


segment_distribution = (
    pd.crosstab(
        df["state"],
        df["district_segment"]
    )
)


segment_distribution["Total Districts"] = (
    segment_distribution.sum(axis=1)
)


segment_distribution["Critical Percentage"] = (
    segment_distribution.get(
        "Critical Priority",
        0
    )
    /
    segment_distribution["Total Districts"]
    * 100
).round(2)


high_risk_states = (
    segment_distribution
    .sort_values(
        "Critical Percentage",
        ascending=False
    )
)


print("\nStates with highest percentage of critical districts")

print(
    high_risk_states[
        [
            "Critical Priority",
            "Total Districts",
            "Critical Percentage"
        ]
    ]
    .head(10)
    .to_string()
)


fig, ax = plt.subplots(
    figsize=(14, 10)
)


heatmap_data = (
    segment_distribution.drop(
        columns=[
            "Total Districts",
            "Critical Percentage"
        ]
    )
)


sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    annot=True,
    fmt="d",
    linewidths=0.5,
    ax=ax,
    cbar_kws={
        "label": "Number of Districts"
    }
)


ax.set_title(
    "Distribution of District Segments Across States",
    fontsize=14,
    fontweight="bold"
)


ax.set_xlabel(
    "District Segment"
)


ax.set_ylabel(
    "State"
)


plt.savefig(
    "outputs/maps/map02_state_segment_heatmap.png"
)


plt.close()


print(
    "SUCCESS: State segment heatmap saved"
)


print("\nCreating welfare vulnerability heatmap")


welfare_heatmap = (
    df.groupby("state")
    [
        [
            "pmkisan_uptake_rate",
            "ab_uptake_rate",
            "bpl_pct",
            "literacy_rate",
            "internet_penetration",
            "development_index"
        ]
    ]
    .mean()
    .round(3)
)


fig, ax = plt.subplots(
    figsize=(12, 9)
)


sns.heatmap(
    welfare_heatmap,
    cmap="RdYlGn",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax,
    cbar_kws={
        "label": "Normalized Score"
    }
)


ax.set_title(
    "State Welfare and Socioeconomic Profile",
    fontsize=14,
    fontweight="bold"
)


ax.set_xlabel(
    "Indicators"
)


ax.set_ylabel(
    "State"
)


plt.savefig(
    "outputs/maps/map03_welfare_profile_heatmap.png"
)


plt.close()


print(
    "SUCCESS: Welfare profile heatmap saved"
)


print("\nClassifying states based on intervention urgency")


def classify_risk(score):

    if score >= 0.45:
        return "Immediate Intervention"

    elif score >= 0.35:
        return "High Monitoring"

    elif score >= 0.25:
        return "Moderate Monitoring"

    else:
        return "Stable"


state_summary["intervention_category"] = (
    state_summary["average_priority_score"]
    .apply(classify_risk)
)


print("\nState Intervention Categories")

print(
    state_summary[
        [
            "average_priority_score",
            "intervention_category"
        ]
    ]
    .to_string()
)


print(
    "SUCCESS: State intervention categories created"
)

print("\nCreating intervention recommendation matrix")


def recommend_action(segment):

    if segment == "Critical Priority":
        return (
            "Immediate outreach, mobile camps, "
            "beneficiary verification, and awareness campaigns"
        )

    elif segment == "Low Coverage":
        return (
            "Increase enrollment drives and improve local service access"
        )

    elif segment == "Medium Coverage":
        return (
            "Maintain monitoring and targeted awareness programs"
        )

    else:
        return (
            "Maintain current strategy and monitor performance"
        )


df["recommended_intervention"] = (
    df["district_segment"]
    .apply(recommend_action)
)


print("SUCCESS: District intervention recommendations generated")


print("\nTop 25 Districts Requiring Immediate Action")


priority_districts = (
    df[
        df["district_segment"] == "Critical Priority"
    ]
    .sort_values(
        "priority_score",
        ascending=False
    )
    .head(25)
)


print(
    priority_districts[
        [
            "district_id",
            "district_name",
            "state",
            "priority_score",
            "pmkisan_uptake_rate",
            "ab_uptake_rate",
            "bpl_pct",
            "literacy_rate",
            "recommended_intervention"
        ]
    ]
    .to_string(index=False)
)


priority_districts.to_csv(
    "data/processed/intervention_priority_districts.csv",
    index=False
)


state_summary.to_csv(
    "data/processed/state_intervention_summary.csv",
    index=True
)


df.to_csv(
    "data/processed/district_data_clean.csv",
    index=False
)


print(
    "\nSUCCESS: Priority district report saved"
)

print(
    "SUCCESS: State intervention summary saved"
)

print(
    "SUCCESS: Updated district dataset saved with recommendations"
)


print("\n" + "=" * 60)
print("STEP 6 COMPLETED: Intervention Mapping Finished")
print("=" * 60)


print(
    "\nGenerated Outputs:"
)

print(
    "MAPS:"
)

print(
    " - map01_high_priority_states.png"
)

print(
    " - map02_state_segment_heatmap.png"
)

print(
    " - map03_welfare_profile_heatmap.png"
)

print(
    "\nDATA FILES:"
)

print(
    " - intervention_priority_districts.csv"
)

print(
    " - state_intervention_summary.csv"
)

print(
    " - district_data_clean.csv (updated)"
)



import os
os.makedirs("outputs/figures", exist_ok=True)

import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

os.makedirs("outputs/figures", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

COLORS = {
    "pmkisan": "#2E86AB",
    "ab": "#E84855",
    "neutral": "#3D405B",
    "good": "#06D6A0",
    "warn": "#FFD166",
    "critical": "#EF476F",
}

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

df = pd.read_csv("data/processed/district_data_clean.csv")

print(f"SUCCESS: Loaded {len(df)} districts for EDA\n")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fig.suptitle(
    "National Distribution of Scheme Uptake Rates",
    fontsize=15,
    fontweight="bold"
)

for ax, (col, label, color) in zip(
    axes,
    [
        ("pmkisan_uptake_rate", "PM-KISAN", COLORS["pmkisan"]),
        ("ab_uptake_rate", "Ayushman Bharat PMJAY", COLORS["ab"]),
    ],
):
    mean_val = df[col].mean()

    ax.hist(
        df[col],
        bins=40,
        color=color,
        alpha=0.75,
        edgecolor="white",
    )

    ax.axvline(
        mean_val,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.1%}",
    )

    ax.axvline(
        0.60,
        color=COLORS["warn"],
        linestyle=":",
        linewidth=1.5,
        label="60% threshold",
    )

    ax.set_xlabel("Uptake Rate (Enrolled / Eligible)")
    ax.set_ylabel("Number of Districts")
    ax.set_title(label)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()

plt.savefig("outputs/figures/fig01_uptake_distribution.png")
plt.close()

print("SUCCESS: Figure 1 - Uptake distribution saved")


state_pivot = (
    df.groupby("state")[["pmkisan_uptake_rate", "ab_uptake_rate"]]
    .mean()
    .sort_values("pmkisan_uptake_rate")
)

state_pivot.columns = [
    "PM-KISAN Uptake",
    "AB PMJAY Uptake",
]

fig, ax = plt.subplots(figsize=(10, 14))

sns.heatmap(
    state_pivot,
    annot=True,
    fmt=".0%",
    cmap="RdYlGn",
    vmin=0.3,
    vmax=0.9,
    ax=ax,
    linewidths=0.5,
    cbar_kws={"label": "Uptake Rate"},
)

ax.set_title(
    "State-Level Average Uptake Rates\n(Sorted by PM-KISAN Uptake)",
    fontsize=13,
    fontweight="bold",
)

ax.set_xlabel("")
ax.set_ylabel("")

plt.savefig("outputs/figures/fig02_state_heatmap.png")
plt.close()

print("SUCCESS: Figure 2 - State heatmap saved")


corr_cols = [
    "pmkisan_uptake_rate",
    "ab_uptake_rate",
    "literacy_rate",
    "female_literacy_rate",
    "rural_pct",
    "bpl_pct",
    "sc_st_pct",
    "internet_penetration",
    "development_index",
    "health_facilities_per_lakh",
    "avg_landholding_ha",
    "gender_literacy_gap",
]

corr_labels = [
    "PM-KISAN\nUptake",
    "AB PMJAY\nUptake",
    "Literacy",
    "Female\nLiteracy",
    "Rural %",
    "BPL %",
    "SC/ST %",
    "Internet\nPenetration",
    "Development\nIndex",
    "Health\nFacilities",
    "Avg Land\n(ha)",
    "Gender\nGap",
]

corr = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(13, 11))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    linewidths=0.4,
    xticklabels=corr_labels,
    yticklabels=corr_labels,
    cbar_kws={"label": "Pearson Correlation"},
)

ax.set_title(
    "Correlation Matrix: Welfare Uptake vs Socioeconomic Indicators",
    fontsize=13,
    fontweight="bold",
)

plt.savefig("outputs/figures/fig03_correlation_matrix.png")
plt.close()

print("SUCCESS: Figure 3 - Correlation matrix saved")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle(
    "Literacy Rate vs Scheme Uptake",
    fontsize=14,
    fontweight="bold",
)

for ax, col, label, color in zip(
    axes,
    ["pmkisan_uptake_rate", "ab_uptake_rate"],
    ["PM-KISAN", "Ayushman Bharat"],
    [COLORS["pmkisan"], COLORS["ab"]],
):

    tier_color = df[col.replace("uptake_rate", "gap_tier")].map(
        {
            "Critical": COLORS["critical"],
            "High": COLORS["warn"],
            "Moderate": "#4CAF50",
            "Low": COLORS["good"],
        }
    )

    ax.scatter(
        df["literacy_rate"],
        df[col],
        c=tier_color,
        alpha=0.5,
        s=18,
        edgecolors="none",
    )

    z = np.polyfit(df["literacy_rate"], df[col], 1)
    p = np.poly1d(z)

    x_line = np.linspace(
        df["literacy_rate"].min(),
        df["literacy_rate"].max(),
        100,
    )

    ax.plot(
        x_line,
        p(x_line),
        color=color,
        linewidth=2,
        label="Trend",
    )


    r = df[["literacy_rate", col]].corr().iloc[0, 1]

    ax.set_title(f"{label} (r = {r:.3f})")
    ax.set_xlabel("Literacy Rate")
    ax.set_ylabel("Uptake Rate")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=COLORS["critical"],
            label="Critical (<40%)",
        ),
        Patch(
            facecolor=COLORS["warn"],
            label="High (40-60%)",
        ),
        Patch(
            facecolor="#4CAF50",
            label="Moderate (60-75%)",
        ),
        Patch(
            facecolor=COLORS["good"],
            label="Low (>75%)",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=8,
    )

plt.savefig(
    "outputs/figures/fig04_literacy_vs_uptake.png"
)

plt.close()

print(
    "SUCCESS: Figure 4 - Literacy versus uptake saved"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)

fig.suptitle(
    "Internet Penetration vs Scheme Uptake",
    fontsize=14,
    fontweight="bold",
)

for ax, col, label, color in zip(
    axes,
    [
        "pmkisan_uptake_rate",
        "ab_uptake_rate",
    ],
    [
        "PM-KISAN",
        "Ayushman Bharat",
    ],
    [
        COLORS["pmkisan"],
        COLORS["ab"],
    ],
):

    ax.scatter(
        df["internet_penetration"],
        df[col],
        alpha=0.4,
        s=16,
        color=color,
        edgecolors="none",
    )

    z = np.polyfit(
        df["internet_penetration"],
        df[col],
        1,
    )

    p = np.poly1d(z)

    x_line = np.linspace(
        df["internet_penetration"].min(),
        df["internet_penetration"].max(),
        100,
    )

    ax.plot(
        x_line,
        p(x_line),
        color="black",
        linewidth=2,
    )

    r = df[
        ["internet_penetration", col]
    ].corr().iloc[0, 1]

    ax.set_title(
        f"{label} (r = {r:.3f})"
    )

    ax.set_xlabel(
        "Internet Penetration"
    )

    ax.set_ylabel(
        "Uptake Rate"
    )

    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

plt.savefig(
    "outputs/figures/fig05_internet_vs_uptake.png"
)

plt.close()

print(
    "SUCCESS: Figure 5 - Internet penetration versus uptake saved"
)    
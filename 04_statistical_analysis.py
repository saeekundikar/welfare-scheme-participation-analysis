import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {
    "pmkisan": "#2E86AB",
    "ab": "#E84855",
    "neutral": "#3D405B",
    "good": "#06D6A0",
    "warn": "#FFD166",
    "critical": "#EF476F",
}

df = pd.read_csv("data/processed/district_data_clean.csv")

print("STEP 4: Statistical Analysis")


results_log = []


print("A. Multiple Linear Regression - Scheme Uptake")


FEATURES = [
    "literacy_rate",
    "internet_penetration",
    "rural_pct",
    "bpl_pct",
    "sc_st_pct",
    "health_facilities_per_lakh",
    "avg_landholding_ha",
    "gender_literacy_gap",
]

for scheme, target in [
    ("PM-KISAN", "pmkisan_uptake_rate"),
    ("Ayushman Bharat", "ab_uptake_rate"),
]:

    X = df[FEATURES].copy()
    y = df[target]

    X_vif = sm.add_constant(X)

    vif_data = pd.DataFrame({
        "feature": FEATURES,
        "VIF": [
            variance_inflation_factor(X_vif.values, i + 1)
            for i in range(len(FEATURES))
        ]
    })

    high_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()

    if high_vif:
        print(f"Removing high VIF features: {high_vif}")
        X = X.drop(columns=high_vif)

    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    print(f"\n{scheme} OLS Regression Results")
    print(
        f"R Squared = {model.rsquared:.4f} | "
        f"Adjusted R Squared = {model.rsquared_adj:.4f}"
    )

    print(
        f"F Statistic = {model.fvalue:.2f} | "
        f"P Value = {model.f_pvalue:.4e}"
    )
    rmse = np.sqrt(
    mean_squared_error(
        y,
        model.fittedvalues
    )
)

    mae = mean_absolute_error(y,model.fittedvalues)

    print(
    f"RMSE = {rmse:.4f} | MAE = {mae:.4f}"
)

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Standard Error": model.bse.values,
        "T Statistic": model.tvalues.values,
        "P Value": model.pvalues.values,
        "Significant": model.pvalues.values < 0.05,
    })

    print("\nCoefficients:")
    print(
        coef_df[
            [
                "Variable",
                "Coefficient",
                "P Value",
                "Significant",
            ]
        ].to_string(index=False)
    )

    df[f"{target.split('_')[0]}_residual"] = model.resid

    results_log.append({
    "scheme": scheme,
    "model": "OLS Regression",
    "r_squared": round(model.rsquared, 4),
    "adj_r_squared": round(model.rsquared_adj, 4),
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "f_statistic": round(model.fvalue, 4),
    "f_pvalue": round(model.f_pvalue, 6),
    "n_observations": int(model.nobs),
})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fig.suptitle(
        f"OLS Regression Diagnostics - {scheme}",
        fontsize=13,
        fontweight="bold",
    )

    predicted_values = model.fittedvalues

    axes[0].scatter(
        y,
        predicted_values,
        alpha=0.3,
        s=12,
        color=COLORS["pmkisan"],
        edgecolors="none",
    )

    limits = [
        min(y.min(), predicted_values.min()),
        max(y.max(), predicted_values.max()),
    ]

    axes[0].plot(
        limits,
        limits,
        "r--",
        linewidth=1.5,
    )

    axes[0].set_xlabel("Actual Uptake Rate")
    axes[0].set_ylabel("Predicted Uptake Rate")
    axes[0].set_title(
        f"Actual vs Predicted (R² = {model.rsquared:.3f})"
    )

    coefficient_plot = (
        coef_df[coef_df["Variable"] != "const"]
        .sort_values("Coefficient")
    )

    bar_colors = [
        COLORS["critical"]
        if value < 0
        else COLORS["good"]
        for value in coefficient_plot["Coefficient"]
    ]

    axes[1].barh(
        coefficient_plot["Variable"],
        coefficient_plot["Coefficient"],
        color=bar_colors,
        alpha=0.85,
    )

    axes[1].axvline(
        0,
        color="black",
        linewidth=1,
    )

    axes[1].set_xlabel("Coefficient")
    axes[1].set_title("Coefficient Magnitudes")

    for index, (_, row) in enumerate(coefficient_plot.iterrows()):

        if row["P Value"] < 0.05:

            offset = 0.002 if row["Coefficient"] > 0 else -0.002

            axes[1].text(
                row["Coefficient"] + offset,
                index,
                "*",
                fontsize=8,
                va="center",
                ha="left" if row["Coefficient"] > 0 else "right",
            )

    plt.tight_layout()

    file_name = (
        scheme.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        f"outputs/figures/fig_stat_{file_name}_regression.png"
    )

    plt.close()

    print("SUCCESS: Regression diagnostic plot saved")

print("\n" + "-" * 60)
print("B. Anomaly Detection - Controlling for Socioeconomic Factors")
print("-" * 60)

print("""
Method: Standardized residuals from OLS regression

Logic:
Expected uptake is estimated based on literacy, internet access,
BPL percentage, and other socioeconomic indicators.

Districts with residuals below -2 standard deviations are considered
underperforming compared to their expected performance.
""")

for target, scheme_name in [
    ("pmkisan_uptake_rate", "pmkisan"),
    ("ab_uptake_rate", "ab"),
]:

    residual_column = f"{scheme_name}_residual"

    if residual_column not in df.columns:
        print(f"WARNING: {residual_column} not found. Skipping analysis.")
        continue

    df[f"{scheme_name}_std_residual"] = (
        (df[residual_column] - df[residual_column].mean())
        / df[residual_column].std()
    )

    df[f"{scheme_name}_anomalous"] = (
        df[f"{scheme_name}_std_residual"] < -2.0
    )


pmkisan_anomalies = (
    df["pmkisan_anomalous"].sum()
    if "pmkisan_anomalous" in df.columns
    else 0
)

ab_anomalies = (
    df["ab_anomalous"].sum()
    if "ab_anomalous" in df.columns
    else 0
)

print(
    f"PM-KISAN anomalous districts: {pmkisan_anomalies}"
)

print(
    f"Ayushman Bharat anomalous districts: {ab_anomalies}"
)


if (
    "pmkisan_anomalous" in df.columns
    and "ab_anomalous" in df.columns
):

    df["dual_anomalous"] = (
        df["pmkisan_anomalous"]
        & df["ab_anomalous"]
    )

    dual_count = df["dual_anomalous"].sum()

    print(
        f"Highest priority districts with both scheme anomalies: {dual_count}"
    )



print("C. Social Disparity Statistical Tests")



median_scst = df["sc_st_pct"].median()

high_scst = df[df["sc_st_pct"] >= median_scst]

low_scst = df[df["sc_st_pct"] < median_scst]


print(
    f"\nSC/ST concentration median split: {median_scst:.1%}"
)


for scheme, column in [
    ("PM-KISAN", "pmkisan_uptake_rate"),
    ("Ayushman Bharat", "ab_uptake_rate"),
]:

    t_statistic, p_value = stats.ttest_ind(
        high_scst[column],
        low_scst[column]
    )

    cohen_d = (
        (high_scst[column].mean() - low_scst[column].mean())
        /
        np.sqrt(
            (
                high_scst[column].std() ** 2
                + low_scst[column].std() ** 2
            ) / 2
        )
    )

    significance = (
        "SIGNIFICANT"
        if p_value < 0.05
        else "NOT SIGNIFICANT"
    )

    print(
        f"\n{scheme}"
    )

    print(
        f"High SC/ST mean: {high_scst[column].mean():.3f}"
    )

    print(
        f"Low SC/ST mean: {low_scst[column].mean():.3f}"
    )

    print(
        f"T Statistic: {t_statistic:.3f}, "
        f"P Value: {p_value:.4e}, "
        f"Cohen's D: {cohen_d:.3f}, "
        f"{significance}"
    )

    results_log.append({
        "scheme": scheme,
        "model": "SC/ST T-Test",
        "t_stat": round(t_statistic, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohen_d, 4),
        "high_scst_mean": round(high_scst[column].mean(), 4),
        "low_scst_mean": round(low_scst[column].mean(), 4),
    })


median_gender_gap = df["gender_literacy_gap"].median()

high_gender_gap = (
    df[df["gender_literacy_gap"] >= median_gender_gap]
)

low_gender_gap = (
    df[df["gender_literacy_gap"] < median_gender_gap]
)


print(
    f"\nGender literacy gap median split: {median_gender_gap:.3f}"
)


for scheme, column in [
    ("PM-KISAN", "pmkisan_uptake_rate"),
    ("Ayushman Bharat", "ab_uptake_rate"),
]:

    t_statistic, p_value = stats.ttest_ind(
        high_gender_gap[column],
        low_gender_gap[column]
    )

    cohen_d = (
        (low_gender_gap[column].mean()
         - high_gender_gap[column].mean())
        /
        np.sqrt(
            (
                high_gender_gap[column].std() ** 2
                + low_gender_gap[column].std() ** 2
            ) / 2
        )
    )

    significance = (
        "SIGNIFICANT"
        if p_value < 0.05
        else "NOT SIGNIFICANT"
    )

    print(
        f"\n{scheme}"
    )

    print(
        f"Low gender gap mean: {low_gender_gap[column].mean():.3f}"
    )

    print(
        f"High gender gap mean: {high_gender_gap[column].mean():.3f}"
    )

    print(
        f"T Statistic: {t_statistic:.3f}, "
        f"P Value: {p_value:.4e}, "
        f"Cohen's D: {cohen_d:.3f}, "
        f"{significance}"
    )

    results_log.append({
        "scheme": scheme,
        "model": "Gender Gap T-Test",
        "t_stat": round(t_statistic, 4),
        "p_value": round(p_value, 6),
        "cohens_d": round(cohen_d, 4),
    })


print("\n" + "-" * 60)
print("D. One-Way ANOVA - Literacy by Gap Tier")
print("-" * 60)


for scheme, tier_column in [
    ("PM-KISAN", "pmkisan_gap_tier"),
    ("Ayushman Bharat", "ab_gap_tier"),
]:

    groups = [
        df[df[tier_column] == level]["literacy_rate"]
        .dropna()
        .values
        for level in ["Critical", "High", "Moderate", "Low"]
    ]

    groups = [group for group in groups if len(group) > 0]

    f_statistic, p_value = stats.f_oneway(*groups)

    overall_mean = df["literacy_rate"].mean()

    ss_between = sum(
        len(group) * (group.mean() - overall_mean) ** 2
        for group in groups
    )

    ss_total = sum(
        (df["literacy_rate"] - overall_mean) ** 2
    )

    eta_squared = ss_between / ss_total


    significance = (
        "SIGNIFICANT"
        if p_value < 0.05
        else "NOT SIGNIFICANT"
    )


    print(
        f"{scheme}: "
        f"F Statistic = {f_statistic:.3f}, "
        f"P Value = {p_value:.4e}, "
        f"Eta Squared = {eta_squared:.4f}, "
        f"{significance}"
    )


    results_log.append({
        "scheme": scheme,
        "model": "ANOVA Literacy by Gap Tier",
        "f_stat": round(f_statistic, 4),
        "p_value": round(p_value, 6),
        "eta_squared": round(eta_squared, 4),
    }) 

if "pmkisan_std_residual" in df.columns:

    fig, ax = plt.subplots(figsize=(11, 7))

    normal_districts = df[
        ~df.get(
            "pmkisan_anomalous",
            pd.Series(False, index=df.index)
        )
    ]

    anomalous_districts = df[
        df.get(
            "pmkisan_anomalous",
            pd.Series(False, index=df.index)
        )
    ]

    ax.scatter(
        normal_districts["literacy_rate"],
        normal_districts["pmkisan_uptake_rate"],
        color="#90CAF9",
        alpha=0.4,
        s=14,
        label="Normal Districts",
        edgecolors="none",
    )

    ax.scatter(
        anomalous_districts["literacy_rate"],
        anomalous_districts["pmkisan_uptake_rate"],
        color=COLORS["critical"],
        alpha=0.9,
        s=40,
        label=f"Anomalous Districts (n={len(anomalous_districts)})",
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_xlabel(
        "Literacy Rate"
    )

    ax.set_ylabel(
        "PM-KISAN Uptake Rate"
    )

    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(1.0)
    )

    ax.set_title(
        "PM-KISAN Anomalous Underperforming Districts",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend()

    plt.savefig(
        "outputs/figures/fig_stat_anomalous_districts.png"
    )

    plt.close()

    print(
        "SUCCESS: Anomaly scatter plot saved"
    )


df.to_csv(
    "data/processed/district_data_clean.csv",
    index=False,
)


pd.DataFrame(results_log).to_csv(
    "data/processed/statistical_results.csv",
    index=False,
)


print(
    "\nSUCCESS: Statistical results saved to data/processed/statistical_results.csv"
)

print(
    "SUCCESS: Updated dataset saved with residuals and anomaly indicators"
)


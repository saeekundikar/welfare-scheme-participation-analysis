import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings

warnings.filterwarnings("ignore")


os.makedirs("outputs/figures", exist_ok=True)
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
    "Critical Priority": "#EF476F",
}


print("=" * 60)
print("STEP 5: District Segmentation Using K-Means Clustering")
print("=" * 60)


df = pd.read_csv(
    "data/processed/district_data_clean.csv"
)



print(
    f"SUCCESS: Loaded {len(df)} districts for segmentation"
)


FEATURES = [
    "pmkisan_uptake_rate",
    "ab_uptake_rate",
    "combined_gap_score",
    "priority_score",
    "literacy_rate",
    "internet_penetration",
    "bpl_pct",
    "rural_pct",
    "development_index",
    "health_facilities_per_lakh",
    "gender_literacy_gap"
]


X = df[FEATURES].copy()


print("\nSelected features for clustering:")

for feature in FEATURES:
    print(f" - {feature}")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


print(
    "\nSUCCESS: Features standardized using StandardScaler"
)


print("\nFinding optimal number of clusters using Elbow Method")


inertia = []

cluster_range = range(2, 11)


for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(
        model.inertia_
    )


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    cluster_range,
    inertia,
    marker="o",
    linewidth=2
)


ax.set_title(
    "Elbow Method for Optimal Number of Clusters",
    fontweight="bold"
)

ax.set_xlabel(
    "Number of Clusters (K)"
)

ax.set_ylabel(
    "Within Cluster Sum of Squares"
)


plt.savefig(
    "outputs/figures/fig_segmentation_elbow_method.png"
)

plt.close()


print(
    "SUCCESS: Elbow method plot saved"
)


OPTIMAL_K = 4


print(
    f"\nSelected number of clusters: {OPTIMAL_K}"
)


print("\nApplying K-Means clustering")


kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)


df["cluster_id"] = kmeans.fit_predict(X_scaled)


print(
    "SUCCESS: K-Means clustering completed"
)
sil_score = silhouette_score(
    X_scaled,
    df["cluster_id"]
)

print(
    f"SUCCESS: Silhouette Score = {sil_score:.4f}"
)


cluster_summary = (
    df.groupby("cluster_id")[
        [
            "pmkisan_uptake_rate",
            "ab_uptake_rate",
            "combined_gap_score",
            "priority_score",
            "literacy_rate",
            "bpl_pct",
            "development_index"
        ]
    ]
    .mean()
    .round(3)
)


print("\nCluster Summary:")
print(cluster_summary)


print(
    "\nAssigning policy labels to clusters"
)


ranking = (
    df.groupby("cluster_id")["priority_score"]
    .mean()
    .sort_values()
)


cluster_labels = {
    ranking.index[0]: "High Coverage",
    ranking.index[1]: "Medium Coverage",
    ranking.index[2]: "Low Coverage",
    ranking.index[3]: "Critical Priority"
}


df["district_segment"] = (
    df["cluster_id"]
    .map(cluster_labels)
)


print(
    "SUCCESS: Cluster labels assigned"
)


print("\nDistrict Distribution by Segment:")

print(
    df["district_segment"]
    .value_counts()
)


print(
    "\nReducing dimensions using PCA for visualization"
)


pca = PCA(
    n_components=2,
    random_state=42
)


components = pca.fit_transform(X_scaled)


df["pca_1"] = components[:, 0]
df["pca_2"] = components[:, 1]


explained_variance = (
    pca.explained_variance_ratio_.sum()
)


print(
    f"SUCCESS: PCA completed with {explained_variance:.2%} variance explained"
)


fig, ax = plt.subplots(
    figsize=(10, 7)
)


for segment, color in COLORS.items():

    subset = df[
        df["district_segment"] == segment
    ]

    ax.scatter(
        subset["pca_1"],
        subset["pca_2"],
        label=segment,
        color=color,
        alpha=0.7,
        s=40,
        edgecolors="black",
        linewidths=0.3
    )


ax.set_title(
    "District Segmentation using K-Means Clustering",
    fontsize=14,
    fontweight="bold"
)


ax.set_xlabel(
    "Principal Component 1"
)


ax.set_ylabel(
    "Principal Component 2"
)


ax.legend(
    title="District Segment"
)


plt.savefig(
    "outputs/figures/fig_segmentation_clusters.png"
)


plt.close()


print(
    "SUCCESS: Cluster visualization saved"
)

print("\n" + "=" * 60)
print("Segment Level Analysis")
print("=" * 60)


segment_statistics = (
    df.groupby("district_segment")
    [
        [
            "pmkisan_uptake_rate",
            "ab_uptake_rate",
            "combined_gap_score",
            "priority_score",
            "literacy_rate",
            "internet_penetration",
            "bpl_pct",
            "rural_pct",
            "development_index",
            "health_facilities_per_lakh"
        ]
    ]
    .agg(
        [
            "mean",
            "min",
            "max" 
        ]
    )
    .round(3)
)

print("\nSegment Statistical Summary:") 
print(segment_statistics.to_string()) 
 

segment_counts = ( 
    df["district_segment"] 
    .value_counts() 
    .rename("district_count") 
) 

print("\nDistricts in Each Segment:") 
print(segment_counts.to_string()) 

critical_districts = ( 
    df[df["district_segment"] == "Critical Priority"] 
    .sort_values( 
        "priority_score", 
        ascending=False 
    ) 
    .head(25) 
) 
print("\nTop 25 Critical Priority Districts:")

print(critical_districts[[
    "district_id",
    "district_name",
    "state",
    "priority_score",
    "pmkisan_uptake_rate",
    "ab_uptake_rate",
    "bpl_pct",
    "literacy_rate"
]].to_string(index=False))


critical_districts.to_csv(
    "data/processed/critical_priority_districts.csv",
    index=False
)

segment_statistics.to_csv(
    "data/processed/segment_statistics.csv"
) 

df.to_csv( 
    "data/processed/district_data_clean.csv",
    index=False
) 

print( 
    "\nSUCCESS: Critical priority districts saved" 
) 

print( 
    "SUCCESS: Segment statistics saved" 
) 

print( 
    "SUCCESS: Updated dataset saved with cluster labels" 
) 
evaluation = pd.DataFrame({
    "metric": [
        "Number of Clusters",
        "Silhouette Score"
    ],
    "value": [
        OPTIMAL_K,
        round(sil_score, 4)
    ]
})

evaluation.to_csv(
    "data/processed/clustering_evaluation.csv",
    index=False
)

print(
    "SUCCESS: Clustering evaluation saved"
)



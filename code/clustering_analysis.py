
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib.colors import ListedColormap

INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
PEACH = "#f4978e" #new color bc need 6 for clusters in case makes 6
GRID = "#e4def5"

BASE = Path(__file__).resolve().parent
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

df = pd.read_csv(BASE / "clean_unlabeled_numeric.csv")

outcome = df["low_birthweight_pct"]
X_df = df.drop(columns=["low_birthweight_pct"])
feature_names= list(X_df.columns)

print("Clustering on", X_df.shape[0], "counties and", X_df.shape[1], "features")

X = StandardScaler().fit_transform(X_df)

#clustering and silhouette score calculation
sil_scores = {}

for k in range(2, 7): 
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(X)
    labels_temp = km.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels_temp)
    print("k =", k, ": silhouette score =", round(sil_scores[k], 3))

best_k = max(sil_scores, key=sil_scores.get)
print("Best k:", best_k)
km_best = KMeans(n_clusters=best_k, n_init=10, random_state=0)
labels = km_best.fit_predict(X)

plt.figure(figsize=(7, 5))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o", color=LILAC)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for K-Means")
plt.axvline(best_k, color=INK, linestyle="dashed")
plt.savefig(IMG / "cluster_silhouette.png")

#pca for visualization
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
palette = [MINT, BLUE, BUTTER, INK, LILAC, PEACH]
plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap=ListedColormap(palette), s=60, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"K-means Clusters (k={best_k}), on PCA Components")
plt.savefig(IMG / "cluster_kmeans_scatter.png", bbox_inches="tight", facecolor="white")

#hierarchical with cosine
cos_dist= pdist(X, metric="cosine")
Z_hier= linkage(cos_dist, method="average")

plt.figure(figsize=(10, 4))
dendrogram(Z_hier, no_labels=True, color_threshold=0, above_threshold_color=BLUE)
plt.title("Hierarchical Clustering Dendrogram (Cosine Distance)")
plt.xlabel("Counties")
plt.ylabel("Distance")
plt.savefig(IMG / "cluster_dendrogram.png", bbox_inches="tight", facecolor="white")

#hierarchical tree at best_k
hclust_labels= fcluster(Z_hier, t=best_k, criterion="maxclust")
print("\nHierarchical cluster counts at k =", best_k)
print(pd.Series(hclust_labels).value_counts().sort_index())

#compare K-Means vs Hierarchical via Crosstab
comparison = pd.crosstab(
    pd.Series(labels, name="Kmeans_cluster"),
    pd.Series(hclust_labels, name="Hierarchical_cluster"))
print("\nCrosstab of K-Means vs. Hierarchical Assignments:")
print(comparison)

#table
from make_figures import table_image, save, GRID, INK, IMG
original_df = pd.read_csv("clean_maternal_infant_health.csv", dtype={"fips": str})
original_df["cluster"] = labels 

cluster_summary = original_df.groupby("cluster").agg(
    County_Count=("fips", "count"),
    Median_Income=("median_income", "median"),
    Child_Poverty=("child_poverty_pct", "mean"),
    Low_Birthweight=("low_birthweight_pct", "mean"),
    Mental_Distress=("mental_distress_pct", "mean")
).reset_index()
cluster_summary["Median_Income"] = cluster_summary["Median_Income"].round(0).astype(int)
cluster_summary["Child_Poverty"] = cluster_summary["Child_Poverty"].round(1)
cluster_summary["Low_Birthweight"] = cluster_summary["Low_Birthweight"].round(1)
cluster_summary["Mental_Distress"] = cluster_summary["Mental_Distress"].round(1)

cluster_summary.columns = [
    "Cluster ID", 
    "County Count", 
    "Median Income ($)", 
    "Child Poverty (%)", 
    "Low Birthweight (%)", 
    "Mental Distress (%)"
]

table_image(
    cluster_summary.astype(str), 
    "Cluster Profiles Summary (Key Indicators)", 
    "cluster_summary_table.png", 
    11
)
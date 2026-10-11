import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
PEACH = "#f4978e"

BASE = Path(__file__).resolve().parent
IMG = BASE / "images"
IMG.mkdir(exist_ok=True) 
df = pd.read_csv(BASE / "clean_unlabeled_numeric.csv")
outcome = df["low_birthweight_pct"]
X_df = df.drop(columns=["low_birthweight_pct"])
feature_names = list(X_df.columns)

from make_figures import table_image

pca_sample_cols = [
    "median_income", "child_poverty_pct", "bachelors_pct", 
    "mental_distress_pct", "depression_pct", "uninsured_pct"]
sample_view = X_df[pca_sample_cols].head(8).copy()

for c in sample_view.select_dtypes("number").columns:
    sample_view[c] = sample_view[c].round(1)

table_image(
    sample_view.astype(str),
    "Sample of features used for PCA (selected columns)",
    "pca_data_sample.png",
    9)

X = StandardScaler().fit_transform(X_df)

print("Running PCA on", X.shape[0], "counties and", X.shape[1], "features")

#pca on all
pca_full = PCA(random_state=0)
pca_full.fit(X)

eigenvalues = pca_full.explained_variance_       
var_ratio = pca_full.explained_variance_ratio_       
cum_var = np.cumsum(var_ratio)

print("\nEigenvalues (first 10):", eigenvalues[:10].round(2))
print("Variance explained by first 10 components:", (var_ratio[:10] * 100).round(1), "%")
print("Cumulative variance through component 10:", (cum_var[9] * 100).round(1), "%")

#k for 90%
k_90 = np.argmax(cum_var >= 0.90) + 1
print("Components needed to explain 90% of variance:", k_90)

#eigenvalues/variance explained per component
fig, ax = plt.subplots(figsize=(8, 5))
n_show = min(20, len(eigenvalues))
ax.plot(range(1, n_show + 1), var_ratio[:n_show] * 100, marker="o", color=MINT)
ax.set_xlabel("Principal component")
ax.set_ylabel("Variance explained (%)")
ax.set_title("Scree Plot: Variance Explained by Each Component")
fig.tight_layout()
fig.savefig(IMG / "pca_scree.png", facecolor="white")

#cumulative variance plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(range(1, n_show + 1), cum_var[:n_show] * 100, marker="o", color=BLUE)
ax.axhline(80, color=INK, linestyle="--", linewidth=1)
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative variance explained (%)")
ax.set_title("Cumulative Variance Explained")
fig.tight_layout()
fig.savefig(IMG / "pca_cumulative.png", facecolor="white")


#reduce to 2 components for visualization
pca2 =PCA(n_components=2, random_state=0)
scores = pca2.fit_transform(X)  

loadings = pd.DataFrame(pca2.components_.T,columns=["PC1", "PC2"],index=feature_names,)
print("\nTop 10 features by absolute loading on PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
print("\nTop 10 features by absolute loading on PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

#counties on PC1 vs PC2
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.5, color=LILAC)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Counties Projected onto the First Two Principal Components")
fig.tight_layout()
fig.savefig(IMG / "pca_scores.png", facecolor="white")

#loadings plot for top 10 
top_n= 10
top_vars = loadings.abs().sum(axis=1).sort_values(ascending=False).head(top_n).index
fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(0, color=PEACH, linewidth=0.8, linestyle="dashed")
ax.axvline(0, color=PEACH, linewidth=0.8, linestyle="dashed")

for var in top_vars:
    x, y = loadings.loc[var, "PC1"], loadings.loc[var, "PC2"]
    ax.arrow(0, 0, x, y, color=BLUE, head_width=0.02, head_length=0.03, length_includes_head=True)
    ax.text(x * 1.3, y * 1.3, var, fontsize=8, color="black", fontweight="bold", ha="center", va="center")

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("PC1 loading")
ax.set_ylabel("PC2 loading")
ax.set_title("Loadings Plot")
fig.tight_layout()
ax.grid(alpha=0.3)
fig.savefig(IMG / "pca_loadings.png", facecolor="white")

#outcome
fig, ax = plt.subplots(figsize=(7, 6))
brand_cmap = LinearSegmentedColormap.from_list(
    "brand_seq", [BUTTER, PEACH, INK])
sc = ax.scatter(scores[:, 0], scores[:, 1], c=outcome, cmap=brand_cmap, s=16, alpha=0.7)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Low birth weight (% of births)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Counties on PC1/PC2, Colored by Low Birth Weight")
fig.tight_layout()
fig.savefig(IMG / "pca_outcome_overlay.png", facecolor="white")

corr_pc1 = np.corrcoef(scores[:, 0], outcome)[0, 1]
corr_pc2 = np.corrcoef(scores[:, 1], outcome)[0, 1]
print("\nCorrelation of PC1 with low birth weight:", round(corr_pc1, 3))
print("Correlation of PC2 with low birth weight:", round(corr_pc2, 3))

#table
from make_figures import table_image, save, GRID, INK, IMG
top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(5)
top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(5)

pca_table_df = pd.DataFrame({
    "Component": ["PC1 (Socioeconomic)"] * 5 + ["PC2 (Demographic)"] * 5,
    "Feature": list(top_pc1.index) + list(top_pc2.index),
    "Absolute Loading": list(top_pc1.values.round(4)) + list(top_pc2.values.round(4))
})

table_image(
    pca_table_df, 
    "Top Principal Component Loadings (PC1 & PC2)", 
    "pca_loadings_table.png", 
    9
)
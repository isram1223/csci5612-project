"""
make_figures.py -- exploratory figures for the DataPrep_EDA page.

Reads the files written by data_prep.py (same folder) and saves the figures
to an images/ folder:
  fig01.png ... fig10.png                  the ten exploration figures
  (histogram, boxplot, heatmap, two scatters, stacked bars, boxplots, race makeup, before/after cleaning)
  raw_cdc_sample.png, raw_census_sample.png, clean_sample.png   data samples

Needs: pandas, numpy, matplotlib
"""

import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BASE = Path(__file__).resolve().parent
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)

# soft palette, matches the website
INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
GRID = "#e4def5"

TIER_ORDER = ["Low_Risk", "Moderate_Risk", "High_Risk"]
TIER_COLORS = {"Low_Risk": MINT, "Moderate_Risk": BUTTER, "High_Risk": BLUE}
TYPE_ORDER = ["Metro", "Micro/Suburban", "Rural"]
TYPE_COLORS = {"Metro": BLUE, "Micro/Suburban": BUTTER, "Rural": MINT}

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

df = pd.read_csv(BASE / "clean_maternal_infant_health.csv", dtype={"fips": str})
df["community_type"] = pd.Categorical(df["community_type"], TYPE_ORDER)
df["infant_risk_tier"] = pd.Categorical(df["infant_risk_tier"], TIER_ORDER)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def read_log():
    # the numbers for figures 7 and 8 come from the cleaning log
    text = (BASE / "cleaning_log.txt").read_text()
    out = {}
    for key, pattern in [
        ("missing", r"Missing values by column: (\{.*\})"),
        ("before", r"Community types before: (\{.*\})"),
        ("after", r"Community types after: (\{.*\})"),
    ]:
        m = re.search(pattern, text)
        out[key] = ast.literal_eval(m.group(1)) if m else None
    return out


log = read_log()

# 1. distribution of the outcome
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["low_birthweight_pct"], bins=35, color=BLUE, edgecolor="white")
med = df["low_birthweight_pct"].median()
ax.axvline(med, color=INK, linestyle="--", linewidth=1.5)
ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.1f}%", color=INK, va="top")
ax.set_title("Distribution of Low Birth Weight Across Counties")
ax.set_xlabel("Low birth weight (% of births)")
ax.set_ylabel("Number of counties")
save(fig, "fig01.png")

# 2. low birth weight by community type
fig, ax = plt.subplots(figsize=(8, 5))
groups = [df.loc[df["community_type"] == t, "low_birthweight_pct"] for t in TYPE_ORDER]
bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                medianprops={"color": INK, "linewidth": 2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, t in zip(bp["boxes"], TYPE_ORDER):
    patch.set_facecolor(TYPE_COLORS[t])
ax.set_xticklabels(TYPE_ORDER)
ax.set_title("Low Birth Weight by Community Type")
ax.set_xlabel("Community type")
ax.set_ylabel("Low birth weight (% of births)")
save(fig, "fig02.png")

# 3. correlation heatmap of the main variables
labels = {
    "low_birthweight_pct": "Low birth weight",
    "child_poverty_pct": "Child poverty",
    "median_income": "Median income",
    "uninsured_pct": "Uninsured",
    "unemployment_pct": "Unemployment",
    "bachelors_pct": "Bachelor's degree",
    "mental_distress_pct": "Frequent mental distress",
    "depression_pct": "Depression",
    "short_sleep_pct": "Short sleep",
    "teen_birth_rate": "Teen birth rate",
    "primary_care_rate": "Primary care doctors",
    "mental_health_provider_rate": "Mental health providers",
    "smoking_pct": "Smoking",
    "inactivity_pct": "Physical inactivity",
    "food_insecurity_pct": "Food insecurity",
    "pm25": "Air pollution (PM2.5)",
}
corr = df[list(labels)].corr()
names = list(labels.values())
cmap = LinearSegmentedColormap.from_list("soft", [BLUE, "#ffffff", LILAC])
fig, ax = plt.subplots(figsize=(10.5, 9))
im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticklabels(names)
ax.grid(False)
for side in ax.spines.values():
    side.set_visible(False)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Correlation")
ax.set_title("Correlation Between Key Community Variables")
save(fig, "fig03.png")


def trend(ax, x, y, label="Trend line"):
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=INK, linewidth=2, label=label)
    return x.corr(y)


def by_type_boxplot(col, ylabel, title, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [df.loc[df["community_type"] == t, col] for t in TYPE_ORDER]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops={"color": INK, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, t in zip(bp["boxes"], TYPE_ORDER):
        patch.set_facecolor(TYPE_COLORS[t])
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_title(title)
    ax.set_xlabel("Community type")
    ax.set_ylabel(ylabel)
    save(fig, name)


# 4. child poverty and low birth weight, colored by community type
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for t in TYPE_ORDER:
    part = df[df["community_type"] == t]
    ax.scatter(part["child_poverty_pct"], part["low_birthweight_pct"], s=14, alpha=0.45,
               color=TYPE_COLORS[t], edgecolors="none", label=t)
r = trend(ax, df["child_poverty_pct"], df["low_birthweight_pct"], "Trend line (all counties)")
ax.legend(title=f"Community type (overall correlation {r:.2f})", frameon=False, loc="upper left")
ax.set_title("Child Poverty vs. Low Birth Weight by Community Type")
ax.set_xlabel("Children living in poverty (%)")
ax.set_ylabel("Low birth weight (% of births)")
save(fig, "fig04.png")

# 5. frequent mental distress and depression
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df["mental_distress_pct"], df["depression_pct"], s=12, alpha=0.35, color=LILAC, edgecolors="none")
r = trend(ax, df["mental_distress_pct"], df["depression_pct"])
ax.legend(title=f"Correlation {r:.2f}", frameon=False, loc="upper left")
ax.set_title("Frequent Mental Distress vs. Depression")
ax.set_xlabel("Frequent mental distress (% of adults)")
ax.set_ylabel("Depression (% of adults)")
save(fig, "fig05.png")

# 6. share of each risk tier within each community type
shares = pd.crosstab(df["community_type"], df["infant_risk_tier"], normalize="index").reindex(TYPE_ORDER)[TIER_ORDER] * 100
fig, ax = plt.subplots(figsize=(8, 5.5))
bottom = np.zeros(len(shares))
for tier in TIER_ORDER:
    vals = shares[tier].values
    bars = ax.bar(shares.index, vals, bottom=bottom, color=TIER_COLORS[tier], label=tier, width=0.6)
    for bar, v, b0 in zip(bars, vals, bottom):
        ax.text(bar.get_x() + bar.get_width() / 2, b0 + v / 2, f"{v:.0f}%", ha="center", va="center", color=INK, fontsize=10)
    bottom += vals
ax.set_ylim(0, 100)
ax.set_title("Infant Risk Tier Proportions by Community Type")
ax.set_xlabel("Community type")
ax.set_ylabel("Share of counties (%)")
ax.legend(title="Infant risk tier", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.grid(axis="x", visible=False)
save(fig, "fig06.png")

# 7. primary care doctors by community type
by_type_boxplot("primary_care_rate", "Primary care doctors (rate from County Health Rankings)",
                "Primary Care Doctor Rate by Community Type", "fig07.png")

# 8. racial and ethnic makeup by community type
groups = {
    "white_pct": "Non-Hispanic\nWhite",
    "black_pct": "Non-Hispanic\nBlack",
    "hispanic_pct": "Hispanic",
    "asian_pct": "Asian",
    "native_american_pct": "American Indian or\nAlaska Native",
}
means = df.groupby("community_type", observed=True)[list(groups)].mean().reindex(TYPE_ORDER)
x = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for i, t in enumerate(TYPE_ORDER):
    bars = ax.bar(x + (i - 1) * 0.26, means.loc[t].values, 0.26, label=t, color=TYPE_COLORS[t])
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(list(groups.values()))
ax.set_ylim(0, means.values.max() * 1.15)
ax.set_title("Average Racial and Ethnic Makeup by Community Type")
ax.set_xlabel("Group")
ax.set_ylabel("Average share of county population (%)")
ax.legend(title="Community type", frameon=False)
ax.grid(axis="x", visible=False)
fig.text(0.01, -0.02, "Groups shown do not add up to 100% (other groups and multiracial residents are not shown).",
         fontsize=8, color=INK)
save(fig, "fig08.png")

# 9. teen birth rate by risk tier
fig, ax = plt.subplots(figsize=(8, 5))
groups9 = [df.loc[df["infant_risk_tier"] == t, "teen_birth_rate"] for t in TIER_ORDER]
bp = ax.boxplot(groups9, patch_artist=True, widths=0.55,
                medianprops={"color": INK, "linewidth": 2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, t in zip(bp["boxes"], TIER_ORDER):
    patch.set_facecolor(TIER_COLORS[t])
ax.set_xticklabels(TIER_ORDER)
ax.set_title("Teen Birth Rate by Infant Risk Tier")
ax.set_xlabel("Infant risk tier")
ax.set_ylabel("Teen births (rate from County Health Rankings)")
save(fig, "fig09.png")

# 10. community types before and after cleaning (from the log)
if log["before"] and log["after"]:
    xt = np.arange(len(TYPE_ORDER))
    before = [log["before"][t] for t in TYPE_ORDER]
    after = [log["after"][t] for t in TYPE_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(xt - 0.2, before, 0.4, label="Before cleaning", color=LILAC)
    b2 = ax.bar(xt + 0.2, after, 0.4, label="After cleaning", color=MINT)
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(xt)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylim(0, max(before + after) * 1.25)
    ax.set_title("Counties by Community Type Before and After Cleaning")
    ax.set_xlabel("Community type")
    ax.set_ylabel("Number of counties")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig10.png")
else:
    print("figure 10 skipped: no community type lines in cleaning_log.txt")


# small images of the raw and clean data
def table_image(frame, title, name, width):
    fig, ax = plt.subplots(figsize=(width, 0.34 * (len(frame) + 2)))
    ax.axis("off")
    tbl = ax.table(cellText=frame.values, colLabels=list(frame.columns), loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(frame.columns))))
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#ece7fb")
            cell.set_text_props(fontweight="bold", color=INK)
        else:
            cell.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    save(fig, name)


raw_cdc = pd.read_csv(BASE / "raw_cdc_data.csv", dtype=str).head(8)
table_image(raw_cdc, "Raw CDC PLACES data (first rows)", "raw_cdc_sample.png", 8)

raw_census = pd.read_csv(BASE / "raw_census_data.csv", dtype=str)
keep = list(raw_census.columns[:5]) + ["state", "county"]
table_image(raw_census[keep].head(8), "Raw Census ACS data (first rows, selected columns)", "raw_census_sample.png", 9)

clean_cols = ["fips", "county", "state", "median_income", "uninsured_pct",
              "mental_distress_pct", "low_birthweight_pct", "community_type", "infant_risk_tier"]
clean_view = df[clean_cols].head(8).copy()
for c in clean_view.select_dtypes("number").columns:
    clean_view[c] = clean_view[c].round(1)
clean_view["median_income"] = clean_view["median_income"].round(0).astype(int)
table_image(clean_view.astype(str), "Cleaned data (first rows, selected columns)", "clean_sample.png", 10)

print("Done. Figures are in", IMG)

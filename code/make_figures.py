"""
make_figures.py: exploratory figures for the DataPrep_EDA page.
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

# match web colors
INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
GRID = "#e4def5"

TIER_ORDER = ["Low_Risk", "Moderate_Risk", "High_Risk"]
TIER_COLORS = {"Low_Risk": GRID, "Moderate_Risk": LILAC, "High_Risk": INK}
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
    #the numbers for figures 7 and 8 come from the cleaning log
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

#distribution of the outcome
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["low_birthweight_pct"], bins=35, color=BLUE, edgecolor="white")
med = df["low_birthweight_pct"].median()
ax.axvline(med, color=INK, linestyle="--", linewidth=1.5)
ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.1f}%", color=INK, va="top")
ax.set_title("Distribution of Low Birth Weight Across Counties")
ax.set_xlabel("Low birth weight (% of births)")
ax.set_ylabel("Number of counties")
save(fig, "fig01.png")

#low birth weight by community type
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

#correlation heatmap of the main variables
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

#Child poverty against low birth weight, by community type
fig, ax = plt.subplots(figsize=(8, 5.5))

#Manually split the data instead of looping, much easier to read
metro = df[df["community_type"] == "Metro"]
micro = df[df["community_type"] == "Micro/Suburban"]
rural = df[df["community_type"] == "Rural"]

ax.scatter(metro["child_poverty_pct"], metro["low_birthweight_pct"], label="Metro", color=BLUE, alpha=0.6)
ax.scatter(micro["child_poverty_pct"], micro["low_birthweight_pct"], label="Micro/Suburban", color=BUTTER, alpha=0.6)
ax.scatter(rural["child_poverty_pct"], rural["low_birthweight_pct"], label="Rural", color=MINT, alpha=0.6)

ax.legend(title="Community Type", frameon=False)
ax.set_title("Child Poverty vs. Low Birth Weight by Community Type")
ax.set_xlabel("Child poverty (%)")
ax.set_ylabel("Low birth weight (% of births)")
save(fig, "fig04.png")

#Frequent mental distress against depression
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df["mental_distress_pct"], df["depression_pct"], alpha=0.4, color=LILAC)

#Simple dropna for the trendline math
clean_df = df.dropna(subset=["mental_distress_pct", "depression_pct"])
if len(clean_df) > 1:
    x = clean_df["mental_distress_pct"]
    y = clean_df["depression_pct"]
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x, slope * x + intercept, color=INK, label="Trend line")
    r = x.corr(y)
    ax.legend(title=f"Correlation {r:.2f}", frameon=False)

ax.set_title("Frequent Mental Distress vs. Depression")
ax.set_xlabel("Frequent mental distress (% of adults)")
ax.set_ylabel("Depression (% of adults)")
save(fig, "fig05.png")

#Risk tier share within each community type
ct = pd.crosstab(df["community_type"], df["infant_risk_tier"], normalize="index") * 100
ct = ct[TIER_ORDER].reindex(TYPE_ORDER)

fig, ax = plt.subplots(figsize=(8, 5.5))
ct.plot(kind="bar", stacked=True, ax=ax, color=[MINT, BUTTER, BLUE], width=0.5)

ax.set_title("Infant Risk Tier Proportions by Community Type")
ax.set_xlabel("Community type")
ax.set_ylabel("Percentage of counties (%)")
ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
save(fig, "fig06.png")

#Access to care by community type
fig, ax = plt.subplots(figsize=(8, 5))
groups = [df.loc[df["community_type"] == t, "primary_care_rate"].dropna() for t in TYPE_ORDER]
bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                medianprops={"color": INK, "linewidth": 2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, t in zip(bp["boxes"], TYPE_ORDER):
    patch.set_facecolor(TYPE_COLORS[t])
ax.set_xticklabels(TYPE_ORDER)
ax.set_title("Primary Care Doctor Rate by Community Type")
ax.set_xlabel("Community type")
ax.set_ylabel("Primary care doctors (rate)")
save(fig, "fig07.png")

#Racial and ethnic makeup by community type
fig, ax = plt.subplots(figsize=(8, 5.5))
race_cols = ["white_pct", "black_pct", "hispanic_pct", "asian_pct", "native_american_pct"]

#group by community type and calculate the mean for the race columns
grouped = df.groupby("community_type")[race_cols].mean().reindex(TYPE_ORDER)

bar_colors = [BLUE, LILAC, MINT, BUTTER, INK]
grouped.plot(kind="bar", ax=ax, width=0.7, color=bar_colors)

ax.set_title("Average Racial and Ethnic Makeup by Community Type")
ax.set_ylabel("Average percentage (%)")
ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
save(fig, "fig08.png")

#Teen birth rate by risk tier
fig, ax = plt.subplots(figsize=(8, 5))
groups = [df.loc[df["infant_risk_tier"] == t, "teen_birth_rate"].dropna() for t in TIER_ORDER]
risk_box_colors = {"Low_Risk": BLUE, "Moderate_Risk": BUTTER, "High_Risk": MINT}
bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                medianprops={"color": INK, "linewidth": 2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, t in zip(bp["boxes"], TIER_ORDER):
    patch.set_facecolor(risk_box_colors[t])
ax.set_xticklabels(TIER_ORDER)
ax.set_title("Teen Birth Rate by Infant Risk Tier")
ax.set_xlabel("Infant risk tier")
ax.set_ylabel("Teen birth rate")
save(fig, "fig09.png")

#Community types before and after cleaning
if log.get("before") and log.get("after"):
    x = np.arange(len(TYPE_ORDER))
    before = [log["before"].get(t, 0) for t in TYPE_ORDER]
    after = [log["after"].get(t, 0) for t in TYPE_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - 0.2, before, 0.4, label="Before cleaning", color=LILAC)
    b2 = ax.bar(x + 0.2, after, 0.4, label="After cleaning", color=MINT)
    ax.bar_label(b1, padding=3)
    ax.bar_label(b2, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_title("Counties by Community Type Before and After Cleaning")
    ax.set_xlabel("Community type")
    ax.set_ylabel("Number of counties")
    ax.set_ylim(0, max(before + after) * 1.25)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig10.png")
else:
    print("figure 10 skipped: no community type lines in cleaning_log.txt")


#raw and clean data images
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
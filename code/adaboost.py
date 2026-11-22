import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from make_figures import table_image

INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
PEACH = "#f4978e"
GRID = "#e4def5"

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

BASE = Path(__file__).resolve().parent
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "clean_labeled_infant_risk.csv")
comm_df = pd.read_csv(BASE / "clean_labeled_community_type.csv")

#add community type
df = df.merge(comm_df[["fips", "community_type"]], on="fips", how="left")

x_raw = df.drop(columns=["infant_risk_tier", "fips"])
x_df = pd.get_dummies(x_raw, columns=["community_type"], dtype=int)
y = df["infant_risk_tier"]

encoded_cols = ["community_type_Metro", "community_type_Micro/Suburban", "community_type_Rural"]
enc_sample = pd.concat([df[["community_type"]].head(8), x_df[encoded_cols].head(8)], axis=1)
table_image(enc_sample.astype(str), "One-Hot Encoding Example", "ada_encoded_sample.png", 9)

#split
split_data = train_test_split(x_df, y, test_size=0.25, random_state=42)
x_train_full = split_data[0]
x_test = split_data[1]
y_train_full = split_data[2]
y_test = split_data[3]

#disjoint
assert set(x_train_full.index).isdisjoint(set(x_test.index))
print("Baseline accuracy:", round(y_test.value_counts(normalize=True).max() * 100, 2), "%")


train_out = x_train_full.copy()
train_out["infant_risk_tier"] = y_train_full
test_out = x_test.copy()
test_out["infant_risk_tier"] = y_test
train_out.to_csv(BASE / "ada_train.csv", index=False)
test_out.to_csv(BASE / "ada_test.csv", index=False)

cols = list(x_df.columns[:6]) + ["infant_risk_tier"]
table_image(train_out[cols].head(8).round(1).astype(str), "AdaBoost Training Set Sample", "ada_train_sample.png", 9)
table_image(test_out[cols].head(8).round(1).astype(str), "AdaBoost Testing Set Sample", "ada_test_sample.png", 9)

split_df = pd.DataFrame({"Dataset Partition": ["Full Data", "Training Set", "Testing Set"],"Counties": [str(len(df)), str(len(x_train_full)), str(len(x_test))],"Percentage": ["100%", "75%", "25%"]})
table_image(split_df, "Train/Test Data Split", "ada_data_split.png", 6)

#valid?
val_split = train_test_split(x_train_full, y_train_full, test_size=0.20, random_state=42)
x_tr = val_split[0]
x_val = val_split[1]
y_tr = val_split[2]
y_val = val_split[3]

learning_rates = [0.05, 0.1, 0.5, 1.0]
colors = [BLUE, MINT, PEACH, LILAC]
best_acc = 0
best_lr = None
best_n = None

fig, ax = plt.subplots(figsize=(8, 5))

for lr, color in zip(learning_rates, colors):
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=150,learning_rate=lr,random_state=42)
    ada.fit(x_tr, y_tr)
    staged_accs = [accuracy_score(y_val, pred) * 100 for pred in ada.staged_predict(x_val)]
    rounds = range(1, 151)
    ax.plot(rounds, staged_accs, label=f"lr = {lr}", color=color, linewidth=2)

    max_idx = int(np.argmax(staged_accs))
    if staged_accs[max_idx] > best_acc:
        best_acc = staged_accs[max_idx]
        best_lr = lr
        best_n = max_idx + 1

print(f"Best validation: lr = {best_lr}, n_estimators = {best_n}, val acc = {best_acc:.2f}%")

ax.set_xlabel("Number of Boosting Rounds (n_estimators)")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("AdaBoost Validation Accuracy as Rounds Are Added")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG / "ada_tuning_curve.png", bbox_inches="tight", facecolor="white")

#model
ada_final = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=150,learning_rate=0.5,random_state=42)
ada_final.fit(x_train_full, y_train_full)
y_pred = ada_final.predict(x_test)
test_acc = accuracy_score(y_test, y_pred) * 100

print("Final AdaBoost test accuracy:", round(test_acc, 4), "%")

cm = confusion_matrix(y_test, y_pred, labels=["Low_Risk", "Moderate_Risk", "High_Risk"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "Moderate", "High Risk"])
brand_cmap = LinearSegmentedColormap.from_list("brand_seq", ["white", BUTTER, PEACH, INK])

fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(cmap=brand_cmap, ax=ax, colorbar=False)
ax.set_title("AdaBoost Confusion Matrix")
ax.grid(False)
fig.tight_layout()
fig.savefig(IMG / "ada_confusion_matrix.png", facecolor="white")
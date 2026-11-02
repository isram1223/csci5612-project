import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from make_figures import table_image
from sklearn.tree import DecisionTreeClassifier

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

x_df = df.drop(columns=["infant_risk_tier", "fips"])
y = df["infant_risk_tier"]
features = list(x_df.columns)
classes = ["High_Risk", "Low_Risk", "Moderate_Risk"] 

split_data = train_test_split(x_df, y, test_size=0.25, random_state=42)
x_train = split_data[0]
x_test = split_data[1]
y_train = split_data[2]
y_test = split_data[3]

#disjoint?
assert set(x_train.index).isdisjoint(set(x_test.index))
print("Baseline guess accuracy:", round(y_test.value_counts(normalize=True).max() * 100, 2), "%")

# save
train_out = x_train.copy()
train_out["infant_risk_tier"] = y_train
test_out = x_test.copy()
test_out["infant_risk_tier"] = y_test
train_out.to_csv(BASE / "dt_train.csv", index=False)
test_out.to_csv(BASE / "dt_test.csv", index=False)


cols = list(x_df.columns[:6]) + ["infant_risk_tier"]
table_image(train_out[cols].head(8).round(1).astype(str), "Training Set Sample", "dt_train_sample.png", 9)
table_image(test_out[cols].head(8).round(1).astype(str), "Testing Set Sample", "dt_test_sample.png", 9)
split_df = pd.DataFrame({"Dataset Partition": ["Full Data", "Training Set", "Testing Set"], "Counties": [str(len(df)), str(len(x_train)), str(len(x_test))],"Percentage": ["100%", "75%", "25%"]})
table_image(split_df, "Train/Test Data Split", "dt_data_split.png", 6)

#model
dt_main = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_main.fit(x_train, y_train)

y_pred = dt_main.predict(x_test)
print("Decision Tree Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred, labels=["Low_Risk", "Moderate_Risk", "High_Risk"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "Moderate", "High Risk"])
brand_cmap = LinearSegmentedColormap.from_list("brand_seq", ["white", MINT, INK])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap=brand_cmap, ax=ax, colorbar=False)
ax.set_title("Decision Tree Confusion Matrix")
ax.grid(False) 
fig.tight_layout()
fig.savefig(IMG / "dt_confusion_matrix.png", facecolor="white")
print("Saved dt_confusion_matrix.png")

#2
dt2 = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=42)
dt2.fit(x_train, y_train)

fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(dt2, 
    feature_names=features, 
    class_names=classes, 
    filled=True, 
    rounded=True, 
    ax=ax, 
    fontsize=12, 
    precision=1)
ax.set_title("Decision Tree (Max Depth = 2)")
fig.tight_layout()
fig.savefig(IMG / "dt_tree_2.png", facecolor="white", dpi=300)
print("Saved dt_tree_2.png")

#3
dt3 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
dt3.fit(x_train, y_train)

fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    dt3, 
    feature_names=features, 
    class_names=classes, 
    filled=True, 
    rounded=True, 
    ax=ax, 
    fontsize=10, 
    precision=1)
ax.set_title("Decision Tree (Max Depth = 3)")
fig.tight_layout()
fig.savefig(IMG / "dt_tree_3.png", facecolor="white", dpi=300)
print("Saved dt_tree_3.png")

#5
dt5 = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt5.fit(x_train, y_train)

fig, ax = plt.subplots(figsize=(42, 16))
plot_tree(
    dt5, 
    feature_names=features, 
    class_names=classes, 
    filled=True, 
    rounded=True, 
    ax=ax, 
    fontsize=8,
    precision=1)
ax.set_title("Decision Tree (Max Depth = 5)")
fig.tight_layout()
fig.savefig(IMG / "dt_tree_5.png", facecolor="white", dpi=150) 
print("Saved dt_tree_5.png")

baseline_acc = y_test.value_counts(normalize=True).max() * 100
print("Baseline guess accuracy:", round(baseline_acc, 2), "%")

for d in [2, 3, 5]:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(x_train, y_train)
    test_acc = dt.score(x_test, y_test) * 100
    train_acc = dt.score(x_train, y_train) * 100
    
    print(f"Max Depth {d} -> Training Accuracy: {round(train_acc, 2)}% | Testing Accuracy: {round(test_acc, 2)}%")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from make_figures import table_image
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

INK = "#3b3350"
BLUE = "#7fb1ea"
MINT = "#7cc7a6"
BUTTER = "#f3c65a"
LILAC = "#a08bd9"
PEACH = "#f4978e" 
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

df = pd.read_csv(BASE / "clean_labeled_infant_risk.csv")
x_df = df.drop(columns=["infant_risk_tier", "fips"])
y = df["infant_risk_tier"]

#split
split_data = train_test_split(x_df, y, test_size=0.25, random_state=42)
x_train = split_data[0]
x_test = split_data[1]
y_train = split_data[2]
y_test = split_data[3]

# disjoint?
assert set(x_train.index).isdisjoint(set(x_test.index))
print("Baseline guess accuracy:", round(y_test.value_counts(normalize=True).max() * 100, 2), "%")

#save
train_out = x_train.copy()
train_out["infant_risk_tier"] = y_train
test_out = x_test.copy()
test_out["infant_risk_tier"] = y_test
train_out.to_csv(BASE / "nb_train.csv", index=False)
test_out.to_csv(BASE / "nb_test.csv", index=False)

cols = list(x_df.columns[:6]) + ["infant_risk_tier"]
table_image(train_out[cols].head(8).round(1).astype(str), "Training Set Sample", "nb_train_sample.png", 9)
table_image(test_out[cols].head(8).round(1).astype(str), "Testing Set Sample", "nb_test_sample.png", 9)

print("Total dataset size:", len(df))
print("Training set:", len(x_train))
print("Testing set:", len(x_test))

split_df = pd.DataFrame({"Dataset Partition": ["Full Data", "Training Set", "Testing Set"],"Counties": [str(len(df)), str(len(x_train)), str(len(x_test))],"Percentage": ["100%", "75%", "25%"]})
table_image(split_df, "Train/Test Data Split", "nb_data_split.png", 6)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

mnb = MultinomialNB()
mnb.fit(x_train_scaled, y_train)
y_pred_mnb = mnb.predict(x_test_scaled)
print("Multinomial Naive Bayes Accuracy:", round(accuracy_score(y_test, y_pred_mnb) * 100, 2), "%")
#matrix 
cm = confusion_matrix(y_test, y_pred_mnb, labels=["Low_Risk", "Moderate_Risk", "High_Risk"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "Moderate", "High Risk"])
brand_cmap = LinearSegmentedColormap.from_list("brand_seq", [BUTTER, PEACH, INK])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap=brand_cmap, ax=ax, colorbar=False)

ax.set_title("Multinomial Naive Bayes Confusion Matrix")
ax.grid(False)
fig.tight_layout()
fig.savefig(IMG / "nb_confusion_matrix.png", facecolor="white")
print("saved nb_confusion_matrix.png")
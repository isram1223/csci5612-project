import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from make_figures import table_image
from sklearn.model_selection import cross_val_score

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

df = pd.read_csv(BASE / "clean_maternal_infant_health.csv")

#binary label
df["high_lbw"] = (df["low_birthweight_pct"] >= 8.5).astype(int)

x_df = df.drop(columns=["fips", "county", "state", "community_type","infant_risk_tier", "low_birthweight_pct", "high_lbw"])
y_lin = df["low_birthweight_pct"]
y_log = df["high_lbw"]
comm_type = df["community_type"]

#split
split_data = train_test_split(x_df, y_lin, y_log, comm_type, test_size=0.25, random_state=42)
x_train = split_data[0]
x_test = split_data[1]
y_lin_train = split_data[2]
y_lin_test = split_data[3]
y_log_train = split_data[4]
y_log_test = split_data[5]
comm_train = split_data[6]
comm_test = split_data[7]
assert set(x_train.index).isdisjoint(set(x_test.index))

train_out = x_train.copy()
train_out["low_birthweight_pct"] = y_lin_train
train_out["high_lbw"] = y_log_train
test_out = x_test.copy()
test_out["low_birthweight_pct"] = y_lin_test
test_out["high_lbw"] = y_log_test

train_out.to_csv(BASE/ "reg_train.csv", index=False)
test_out.to_csv(BASE/ "reg_test.csv", index=False)

cols = list(x_df.columns[:5]) + ["low_birthweight_pct", "high_lbw"]
table_image(train_out[cols].head(8).round(1).astype(str), "Regression Training Set Sample", "reg_train_sample.png", 9)
table_image(test_out[cols].head(8).round(1).astype(str), "Regression Testing Set Sample", "reg_test_sample.png", 9)

split_df = pd.DataFrame({"Dataset Partition": ["Full Data", "Training Set", "Testing Set"],"Counties": [str(len(df)), str(len(x_train)), str(len(x_test))],"Percentage": ["100%", "75%", "25%"]})
table_image(split_df, "Train/Test Data Split", "reg_data_split.png", 6)

#linear regression
lin_reg= LinearRegression()
lin_reg.fit(x_train, y_lin_train)
y_lin_pred = lin_reg.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_lin_test, y_lin_pred))
mae = mean_absolute_error(y_lin_test, y_lin_pred)
r2 = r2_score(y_lin_test, y_lin_pred)

print("Linear RMSE:", round(rmse, 4), "MAE:", round(mae, 4), "R2:", round(r2, 4))

lin_coefs =pd.Series(lin_reg.coef_, index=x_df.columns)
key_vars = ["child_poverty_pct", "median_income", "income_ratio","mental_distress_pct", "teen_birth_rate", "smoking_pct","uninsured_pct", "bachelors_pct"]
print("Linear Coefficients:")
print(lin_coefs[key_vars].round(4))

#predicted vs actual (community type)
fig, ax = plt.subplots(figsize=(7, 6))
metro =comm_test == "Metro"
micro = comm_test == "Micro/Suburban"
rural = comm_test == "Rural"
ax.scatter(y_lin_test[metro], y_lin_pred[metro], label="Metro", color=BLUE, alpha=0.65, edgecolors=INK, linewidths=0.3)
ax.scatter(y_lin_test[micro], y_lin_pred[micro], label="Micro/Suburban", color=BUTTER, alpha=0.7, edgecolors=INK, linewidths=0.3)
ax.scatter(y_lin_test[rural], y_lin_pred[rural], label="Rural", color=MINT, alpha=0.7, edgecolors=INK, linewidths=0.3)

min_val = min(y_lin_test.min(), y_lin_pred.min())
max_val = max(y_lin_test.max(), y_lin_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=INK, linewidth=2, label="Perfect Prediction")

ax.set_xlabel("Actual Low Birth Weight (%)")
ax.set_ylabel("Predicted Low Birth Weight (%)")
ax.set_title("Linear Regression: Predicted vs. Actual by Community Type")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(IMG / "reg_pred_vs_actual.png", bbox_inches="tight", facecolor="white")

#logistic regression
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#choose C using 5-fold cross-validation on training set
c_values = [0.01, 0.1, 1.0, 10.0]
best_c = 1.0
best_cv = 0
for c in c_values:
    cv_score = cross_val_score(LogisticRegression(C=c, max_iter=1000, random_state=42), x_train_scaled, y_log_train, cv=5).mean() * 100
    print(f"C = {c} | 5-fold CV acc = {cv_score:.2f}%")
    if cv_score > best_cv:
        best_cv = cv_score
        best_c = c

log_reg= LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(x_train_scaled, y_log_train)
y_log_pred = log_reg.predict(x_test_scaled)
log_acc= accuracy_score(y_log_test, y_log_pred) * 100
base_acc = y_log_test.value_counts(normalize=True).max() * 100
print("Binary Baseline:", round(base_acc, 2), "% | Logistic Acc:", round(log_acc, 2), "%")

log_coefs = pd.DataFrame({"coef": log_reg.coef_[0], "odds_ratio": np.exp(log_reg.coef_[0])}, index=x_df.columns)
print("Logistic Coefficients & Odds Ratios:")
print(log_coefs.loc[key_vars].round(4))

cm =confusion_matrix(y_log_test, y_log_pred, labels=[0, 1])
disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below 8.5%", "High Risk (>=8.5%)"])
brand_cmap= LinearSegmentedColormap.from_list("brand_seq", ["white", LILAC, INK])

fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(cmap=brand_cmap, ax=ax, colorbar=False)
ax.set_title("Logistic Regression Confusion Matrix")
ax.grid(False)
fig.tight_layout()
fig.savefig(IMG / "reg_confusion_matrix.png", facecolor="white")

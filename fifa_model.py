import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

np.random.seed(42)

df = pd.read_csv("world_cup_training.csv")

FEATURES = [
    "fifa_rank",
    "fifa_points",
    "confederation_code",
    "wc_appearances",
    "prev_wc_wins",
    "is_host",
]

TARGET = "won"

X = df[FEATURES]
y = df[TARGET]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

logo = LeaveOneGroupOut()
groups = df["year"]
cv_score = cross_val_score(model, X, y, cv=logo, groups=groups, scoring="roc_auc")

print(f"Cross-validation:")
for year, score in zip(sorted(df["year"].unique()), cv_score):
    bar = "|" * int(score * 20)
    print(f"Held out {year}: AUC = {score:.3f} {bar}")
print(f"Mean AUC: {cv_score.mean():.3f}")

model.fit(X, y)

df_2026 = pd.read_csv("world_cup_2026.csv")

X_2026 = df_2026[FEATURES]
df_2026["win_probability"] = model.predict_proba(X_2026)[:, 1]

total = df_2026["win_probability"].sum()
df_2026["win_probability_pct"] = (df_2026["win_probability"] / total * 100).round(1)

result = df_2026[["team", "fifa_rank", "win_probability_pct"]].sort_values(
    "win_probability_pct", ascending=False
)

print("2026 WORLD CUP - AI WIN PREDICTION")
print(f"{'Team':<20}  {'Rank':<6}  {'Probability'}")

for _, row in result.head(10).iterrows():
    bar = "|" * int(row["win_probability_pct"] / 2)
    print(
        f"{row['team']:<20}  #{int(row['fifa_rank']):<5} {bar} {row['win_probability_pct']:.1f}%"
    )

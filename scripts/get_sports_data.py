"""Download small, leakage-free F1 and football datasets into Desktop/testdataset.

Both predict genuinely uncertain outcomes (no perfect-accuracy giveaways):
- F1:       will a car finish in the points (top 10)?  -> from pre-race info only
- Football: home win / draw / away win (FTR)            -> from pre-match odds only
"""
import sys
from pathlib import Path

import pandas as pd

# Output dir: first CLI arg if given, else a repo-relative ./data folder.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- Formula 1: points-finish (top 10) ----------------
F1 = "https://raw.githubusercontent.com/toUpperCase78/formula1-datasets/master/"
f1_files = {
    2022: "Formula1_2022season_raceResults.csv",
    2023: "Formula1_2023season_raceResults.csv",
    2024: "Formula1_2024season_raceResults.csv",
}
frames = []
for year, fname in f1_files.items():
    try:
        d = pd.read_csv(F1 + fname)
        pos = pd.to_numeric(d["Position"], errors="coerce")  # "NC"/"DNF"/"DQ" -> NaN
        frames.append(pd.DataFrame({
            "Season": year,
            "Track": d["Track"],
            "Driver": d["Driver"],
            "Team": d["Team"],
            "StartingGrid": pd.to_numeric(d["Starting Grid"], errors="coerce"),
            "PointsFinish": (pos <= 10).astype(int),  # top 10 = scored points
        }))
        print(f"F1 {year}: {len(d)} rows")
    except Exception as e:
        print(f"F1 {year} skipped: {e}")

f1 = pd.concat(frames, ignore_index=True)
f1.to_csv(OUT / "f1_points_finish.csv", index=False)
print(f"-> f1_points_finish.csv {f1.shape} | points-finish rate {f1['PointsFinish'].mean():.2f}\n")

# ---------------- Football (EPL): match result from odds ----------------
keep = ["HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "FTR"]
frames = []
for season in ["2122", "2223", "2324"]:
    try:
        d = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv")
        frames.append(d[keep].dropna(subset=["FTR"]))
        print(f"EPL {season}: {len(d)} matches")
    except Exception as e:
        print(f"EPL {season} skipped: {e}")

epl = pd.concat(frames, ignore_index=True)
epl.to_csv(OUT / "football_epl_results.csv", index=False)
print(f"-> football_epl_results.csv {epl.shape}")
print("EPL outcome balance:", epl["FTR"].value_counts(normalize=True).round(3).to_dict())

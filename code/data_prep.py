"""
Sources
    CDC PLACES (Socrata API): frequent mental distress, short sleep, depression
    Census ACS 5-year profile (API): income, education, housing, age
    County Health Rankings 2025: analytic_data_2025.csv
    USDA Rural-Urban Continuum Codes 2023: rural_urban_codes_2023.csv

python data_prep.py = runs everything
python data_prep.py labels = prints what each Census variable code means
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests

#the folder this script sits in; inputs are read from here, outputs go here
BASE = Path(__file__).resolve().parent

TARGET = "low_birthweight_pct"

#paste your Census key between the quotes, do not post publicly 
CENSUS_KEY = ""
MAX_MISSING = 0.15  #drop a column if more than 15% of its values are missing

#Census profile variables (2022 ACS 5-year)
#python data_prep.py labels to confirm 
ACS_VARS = {
    "DP05_0001E": "total_population",
    "DP05_0018E": "median_age",
    "DP05_0019PE": "under_18_pct",
    "DP05_0024PE": "over_64_pct",
    "DP02_0038PE": "single_mother_pct",
    "DP02_0065PE": "bachelors_pct",
    "DP02_0072PE": "disability_pct",
    "DP03_0009PE": "unemployment_pct",
    "DP03_0062E": "median_income",
    "DP03_0099PE": "uninsured_pct",
    "DP04_0003PE": "vacant_housing_pct",
    "DP04_0047PE": "renter_pct",
    "DP04_0058PE": "no_vehicle_pct",
    "DP04_0134E": "median_rent",
}

#County Health Rankings measures: name in file -> (new name, multiplier).
#Stored as props so multiply by 100
CHR_MEASURES = {
    "Low Birth Weight": ("low_birthweight_pct", 100),
    "Children in Poverty": ("child_poverty_pct", 100),
    "Adult Smoking": ("smoking_pct", 100),
    "Adult Obesity": ("obesity_pct", 100),
    "Physical Inactivity": ("inactivity_pct", 100),
    "Excessive Drinking": ("excessive_drinking_pct", 100),
    "Food Insecurity": ("food_insecurity_pct", 100),
    "Some College": ("some_college_pct", 100),
    "Teen Births": ("teen_birth_rate", 1),
    "Mental Health Providers": ("mental_health_provider_rate", 1),
    "Primary Care Physicians": ("primary_care_rate", 1),
    "Income Inequality": ("income_ratio", 1),
    "Air Pollution: Particulate Matter": ("pm25", 1),
    "Poor Mental Health Days": ("poor_mental_health_days", 1),
    "% Non-Hispanic Black": ("black_pct", 100),
    "% Hispanic": ("hispanic_pct", 100),
    "% Asian": ("asian_pct", 100),
    "% American Indian or Alaska Native": ("native_american_pct", 100),
    "% Non-Hispanic White": ("white_pct", 100),
}

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def save_sample(records, filename, n=10):
    with open(BASE / filename, "w") as f:
        json.dump(records[:n], f, indent=2)


def check_labels():
    base = "https://api.census.gov/data/2022/acs/acs5/profile/variables/"
    for code, name in ACS_VARS.items():
        r = requests.get(base + code + ".json", timeout=30)
        label = r.json().get("label", "?") if r.ok else "not found"
        print(f"{code:12} {name:20} {label}")


def get_cdc():
    url = "https://chronicdata.cdc.gov/resource/swc5-untb.json"
    params = {
        "$select": "year,stateabbr,locationname,locationid,measureid,data_value",
        "$where": "measureid in('MHLTH','SLEEP','DEPRESSION') and datavaluetypeid='CrdPrv'",
        "$limit": 50000,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    records = r.json()
    save_sample(records, "raw_cdc_api.json")

    raw = pd.DataFrame(records)
    raw.to_csv(BASE / "raw_cdc_data.csv", index=False)
    log(f"CDC: {len(raw)} raw rows")

    df = raw.copy()
    if "year" in df.columns:
        log(f"CDC: release years in pull: {sorted(df['year'].unique())}")
        #keep the most recent year available for each county and measure
        df = df.sort_values("year").drop_duplicates(
            ["locationid", "measureid"], keep="last"
        )
    df["fips"] = df["locationid"].astype(str).str.zfill(5)
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    if df.duplicated(["fips", "measureid"]).any():
        log("CDC: duplicate county/measure rows found, keeping the first")

    wide = df.pivot_table(
        index="fips", columns="measureid", values="data_value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "MHLTH": "mental_distress_pct",
            "SLEEP": "short_sleep_pct",
            "DEPRESSION": "depression_pct",
        }
    )
    log(f"CDC: {len(wide)} counties after reshaping")
    return wide


def load_key():
    #order: key pasted at the top of this file
    #if not check a file called census_key.txt
    key = CENSUS_KEY or os.environ.get("CENSUS_API_KEY")
    key_file = BASE / "census_key.txt"
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key:
        raise SystemExit("No Census key found. Paste it into CENSUS_KEY at the top of data_prep.py.")
    return key


def fetch_census(key):
    url = "https://api.census.gov/data/2022/acs/acs5/profile"
    params = {
        "get": "NAME," + ",".join(ACS_VARS),
        "for": "county:*",
        "in": "state:*",
        "key": key,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    try:
        rows = r.json()
    except ValueError:
        #a bad or inactive key comes back as a plain message, not json
        raise ValueError(f"Census said: {r.text[:200]}")
    save_sample(rows, "raw_census_api.json")
    raw = pd.DataFrame(rows[1:], columns=rows[0])
    raw.to_csv(BASE / "raw_census_data.csv", index=False)
    return raw

def get_census():
    raw = fetch_census(load_key())
    log(f"Census: {len(raw)} raw rows")

    raw["fips"] = raw["state"].str.zfill(2) + raw["county"].str.zfill(3)
    nums = raw[list(ACS_VARS)].apply(pd.to_numeric, errors="coerce")

    #the API uses values like -666666666 when an estimate is suppressed
    suppressed = (nums < -1e6).sum().sum()
    nums = nums.where(nums > -1e6).rename(columns=ACS_VARS)
    log(f"Census: {suppressed} suppressed values set to NaN")

    return pd.concat([raw[["fips"]], nums], axis=1)


def get_chr():
    path = BASE / "analytic_data_2025.csv"
    raw = pd.read_csv(path, skiprows=[1], low_memory=False)
    out = pd.DataFrame(
        {
            "fips": raw["5-digit FIPS Code"].astype(str).str.zfill(5),
            "county": raw["Name"],
            "state": raw["State Abbreviation"],
        }
    )
    for measure, (new_name, scale) in CHR_MEASURES.items():
        col = f"{measure} raw value"
        if col not in raw.columns:
            log(f"CHR: column '{col}' not in file, skipped")
            continue
        out[new_name] = pd.to_numeric(raw[col], errors="coerce") * scale
    log(f"CHR: {len(out)} rows, {out.shape[1] - 3} measures kept")
    return out


def get_usda():
    path = BASE / "rural_urban_codes_2023.csv"
    raw = pd.read_csv(path, encoding="latin1")
    df = raw[raw["Attribute"] == "RUCC_2023"].copy()
    df["fips"] = df["FIPS"].astype(str).str.zfill(5)
    df["rucc_code"] = pd.to_numeric(df["Value"], errors="coerce")

    def group(code):
        if code <= 3:
            return "Metro"
        if code <= 6:
            return "Micro/Suburban"
        return "Rural"

    df = df.dropna(subset=["rucc_code"])
    df["community_type"] = df["rucc_code"].apply(group)
    out = df[["fips", "community_type"]].drop_duplicates("fips")
    log(f"USDA: {len(out)} counties")
    return out


def build_clean(census, cdc, chr_df, usda):
    df = census.copy()
    for name, other in [("CDC", cdc), ("CHR", chr_df), ("USDA", usda)]:
        before = len(df)
        df = df.merge(other, on="fips", how="inner")
        log(f"Merge with {name}: {before} -> {len(df)} rows")

    #state totals (fips ending in 000) are not counties
    df = df[~df["fips"].str.endswith("000")].reset_index(drop=True)
    log(f"After removing state totals: {len(df)} rows")

    #make sure percentage between 0 and 100 (impossible otherwise)
    for col in [c for c in df.columns if c.endswith("_pct")]:
        bad = df[col].notna() & ~df[col].between(0, 100)
        if bad.any():
            log(f"Range check: {bad.sum()} bad values in {col} set to NaN")
            df.loc[bad, col] = np.nan

    #drop columns that are mostly empty, then rows that still have gaps
    missing = df.isna().mean()
    too_empty = [c for c in missing[missing > MAX_MISSING].index if c != TARGET]
    if too_empty:
        log(f"Dropping columns with >{MAX_MISSING:.0%} missing: {too_empty}")
        df = df.drop(columns=too_empty)
    gaps = df.isna().sum()
    log(f"Missing values by column: {gaps[gaps > 0].to_dict()}")
    log(f"Community types before: {df['community_type'].value_counts().to_dict()}")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    log(f"Dropped {before - len(df)} rows with missing values, {len(df)} remain")
    log(f"Community types after: {df['community_type'].value_counts().to_dict()}")

    #3-tier label from low birth weight, using classification 
    df["infant_risk_tier"] = pd.qcut(
        df[TARGET], q=3, labels=["Low_Risk", "Moderate_Risk", "High_Risk"]
    )
    return df


def save_versions(df):
    df.to_csv(BASE / "clean_maternal_infant_health.csv", index=False)

    #numeric columns only, for clustering and PCA
    df.select_dtypes("number").to_csv(BASE / "clean_unlabeled_numeric.csv", index=False)

    #the label is built from low birth weight, so it has to leave the features
    risk = df.drop(columns=["county", "state", "community_type", TARGET])
    risk.to_csv(BASE / "clean_labeled_infant_risk.csv", index=False)

    community = df.drop(columns=["county", "state", "infant_risk_tier"])
    community.to_csv(BASE / "clean_labeled_community_type.csv", index=False)

    log(f"Saved clean files. Full dataset shape: {df.shape}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "labels":
        check_labels()
        sys.exit()

    cdc = get_cdc()
    census = get_census()
    chr_df = get_chr()
    usda = get_usda()

    final = build_clean(census, cdc, chr_df, usda)
    save_versions(final)

    with open(BASE / "cleaning_log.txt", "w") as f:
        f.write("\n".join(log_lines))
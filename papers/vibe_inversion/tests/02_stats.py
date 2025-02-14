import os
import sys
from pathlib import Path

import pandas as pd

# Load MEVIBE dataset
# /run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_processed/UKBB/info
epi = pd.read_excel("/media/data/NAKO/notes/NAKO_epidemiological.xlsx")
epi["Subject_Name"] = epi["Subject_Name"].astype(int)  # Ensure `Subject_Name` is int
epi["BMI"] = epi["PatientWeight"] / epi["PatientSize"] ** 2
epi["BMI_Class"] = pd.cut(
    epi["BMI"],
    bins=[0, 18.5, 25, 30, 35, 40, float("inf")],
    labels=["Underweight", "Healthy Weight", "Overweight", "Class 1 Obesity", "Class 2 Obesity", "Class 3 Obesity"],
)
for name, bl, percent in [
    ("NAKO MEVIBE", Path(os.environ.get("DATASET_NAKO", ""), "notes/water_fat_inversion_mevibe.xlsx"), 0.001),
    ("NAKO VIBE", Path(os.environ.get("DATASET_NAKO", ""), "notes/water_fat_inversion_vibe.xlsx"), 0.001),  # vibe are stichted from 4
]:
    # Load blacklist and filter
    bl_df = pd.read_excel(bl)
    # Convert types to match
    bl_df["sub"] = bl_df["sub"].str.split("_").str[0].astype(int)  # Convert blacklist `sub` to int
    bl_df = bl_df[bl_df["percent"] > percent]
    black_list = bl_df["sub"].to_numpy()
    # print(black_list)
    # Add BMI and BMI classification

    def fun(x):
        return 1 if x in black_list else 0

    # Add `has_error` column
    epi["has_error"] = epi["Subject_Name"].apply(fun)
    for agg in ["BMI_Class", "InstitutionAddress", "PatientSex"]:  # "PatientAge", "PatientSize", "PatientWeight"
        # Statistics over BMI classes
        error_stats = epi.groupby(agg)["has_error"].agg(["sum", "count"])
        error_stats["percent_error"] = (error_stats["sum"] / error_stats["count"]) * 100
        # Print statistics summary
        print()
        print(f"{name}: Atleast {percent * 100} % of the image has a swap:")
        print(error_stats)

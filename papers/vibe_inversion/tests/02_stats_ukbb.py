from pathlib import Path

import pandas as pd

# Load MEVIBE dataset
epi = pd.read_csv("/run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_processed/UKBB/info/basic_features.csv")
epi["Subject_Name"] = epi["eid"].astype("Int64")  # Allow missing values
epi["PatientSex"] = epi["31-0.0"]
epi["BMI"] = epi["21001-2.0"]
epi["BMI_Class"] = pd.cut(
    epi["BMI"],
    bins=[0, 18.5, 25, 30, 35, 40, float("inf")],
    labels=["Underweight", "Healthy Weight", "Overweight", "Class 1 Obesity", "Class 2 Obesity", "Class 3 Obesity"],
)


percent = 0.001
for name, bl in [
    ("UKBB VIBE", Path("/run/user/1000/gvfs/smb-share:server=172.21.251.64,share=nas/datasets_processed/UKBB/info/water-fat-swap.xlsx")),
]:
    # Load blacklist and filter
    bl_df = pd.read_excel(bl)
    bl_df["sub"] = bl_df["sub"].astype(int)  # Convert blacklist `sub` to int
    bl_df["swap_size"] = pd.cut(
        bl_df["percent"],
        bins=[0, 0.001, 0.01, 0.1, 0.5, 1],
        labels=["0.1 %", "0.1 - 1 %", "1 - 10 %", "10 - 50 %", "50 - 100 %"],
    )
    bl_df["one"] = 1
    print(bl_df.groupby("swap_size")["one"].agg(["sum", "count"]))

    bl_df = bl_df[bl_df["percent"] > percent]
    black_list = bl_df["sub"].to_numpy()

    # Assign "unknown" to missing Subject_Name
    epi["Subject_Name"] = epi["Subject_Name"].fillna("unknown")

    # Combine all unique identifiers
    all_subjects = set(epi["Subject_Name"].unique()).union(black_list)

    # Create a new DataFrame to ensure all subjects are accounted for
    combined_df = pd.DataFrame({"Subject_Name": list(all_subjects)})
    combined_df["Subject_Name"] = combined_df["Subject_Name"].astype("Int64", errors="ignore")
    combined_df = combined_df.merge(epi, on="Subject_Name", how="left")

    # Mark errors based on blacklist
    combined_df["has_error"] = combined_df["Subject_Name"].apply(lambda x: 1 if x in black_list else 0)

    # Assign "unknown" to missing group values (e.g., BMI_Class, PatientSex)
    combined_df["BMI_Class"] = combined_df["BMI_Class"].cat.add_categories("unknown").fillna("unknown")
    combined_df["PatientSex"] = combined_df["PatientSex"].fillna("unknown")

    for agg in ["BMI_Class", "PatientSex"]:  # Add other columns as needed
        # Statistics over groups
        error_stats = combined_df.groupby(agg)["has_error"].agg(["sum", "count"])
        error_stats["percent_error"] = (error_stats["sum"] / error_stats["count"]) * 100

        # Print statistics summary
        print()
        print(f"{name}: At least {percent * 100}% of the image has a swap:")
        print(error_stats)

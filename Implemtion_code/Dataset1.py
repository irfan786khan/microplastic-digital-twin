import os
import pandas as pd
from datasets import load_dataset

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

SAVE_DIR = "./data/huggingface"
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 80)
print("TARGETED DOWNLOAD FOR MICROPLASTIC DIGITAL TWIN RESEARCH")
print("=" * 80)



# These datasets contain water quality, climate, and environmental data
datasets_to_download = [
    {
        'name': 'Antajitters/WindSpeed_10m',
        'description': 'Wind speed at 10m height - climate data',
        'columns': ['Date & Time Stamp', 'SpeedAvg', 'SpeedMax', 'DirectionAvg', 'TemperatureAvg']
    },
    {
        'name': 'Antajitters/WindSpeed_50m',
        'description': 'Wind speed at 50m height - climate data',
        'columns': ['Date & Time Stamp', 'Speed Avg 10m', 'SpeedMax', 'DirectionAvg', 'TemperatureAvg']
    },
    {
        'name': 'Antajitters/WindSpeed_100m',
        'description': 'Wind speed at 100m height - climate data',
        'columns': ['Date & Time Stamp', 'Speed Avg 10m', 'SpeedMax', 'DirectionAvg', 'TemperatureAvg']
    },
    {
        'name': '2imi9/OlmoEarth-v1-FT-Karst-Groundwater-Base',
        'description': 'Groundwater data - hydrology',
        'columns': ['category', 'tag', 'lon', 'lat', 'oe_start_time']
    }
]


downloaded = 0
failed = 0
dataset_info = []

print("\n📥 DOWNLOADING TARGETED DATASETS")
print("=" * 80)

for ds_info in datasets_to_download:
    dataset_name = ds_info['name']
    description = ds_info['description']

    print(f"\n📦 Dataset: {dataset_name}")
    print(f"   Description: {description}")

    try:
        # Load dataset
        dataset = load_dataset(dataset_name, trust_remote_code=True)

        # Combine all splits
        all_data = []
        for split in dataset.keys():
            try:
                df_split = dataset[split].to_pandas()
                all_data.append(df_split)
                print(f"   📊 Split '{split}': {len(df_split)} rows")
            except Exception as e:
                print(f"   ⚠️ Cannot load split '{split}': {str(e)[:50]}")
                continue

        # Combine and save
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates()

            filename = dataset_name.replace("/", "_")
            save_path = os.path.join(SAVE_DIR, f"{filename}.csv")

            combined_df.to_csv(save_path, index=False)

            dataset_info.append({
                'dataset': dataset_name,
                'description': description,
                'rows': len(combined_df),
                'columns': len(combined_df.columns),
                'column_names': list(combined_df.columns),
                'path': save_path
            })

            print(f"   ✅ SAVED: {os.path.basename(save_path)}")
            print(f"      Total rows: {len(combined_df):,}")
            print(f"      Total columns: {len(combined_df.columns)}")
            print(f"      Columns: {list(combined_df.columns)[:5]}...")

            downloaded += 1
        else:
            print(f"   ❌ No data loaded")
            failed += 1

    except Exception as e:
        failed += 1
        print(f"   ❌ Failed: {str(e)[:100]}")


print("\n" + "=" * 80)
print("✅ DOWNLOAD COMPLETE")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"   Downloaded: {downloaded} datasets")
print(f"   Failed: {failed} datasets")

print("\n📁 Downloaded datasets:\n")
for info in dataset_info:
    print(f"   📄 {os.path.basename(info['path'])}")
    print(f"      Description: {info['description']}")
    print(f"      Rows: {info['rows']:,}, Columns: {info['columns']}")
    print(f"      Columns: {', '.join(info['column_names'][:5])}...")
    print()



if dataset_info:
    index_df = pd.DataFrame(dataset_info)
    index_df.to_csv(os.path.join(SAVE_DIR, "dataset_index.csv"), index=False)
    print(f"📊 Dataset index saved: dataset_index.csv")

